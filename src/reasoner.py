import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics

from src.simplefact import *
from src.simplefact.syntax import *
from src.utils import *
from src.vis import *

def core(expr):
	if isinstance(expr, int) or expr == TOP or expr == BOT:
		return expr
	if expr[0] == NOT:
		inner = expr[1]
		if inner == TOP:
			return BOT
		if inner == BOT:
			return TOP
		if isinstance(inner, tuple):
			if inner[0] == NOT:
				return core(inner[1])
			if inner[0] == OR:
				return AND, core((NOT, inner[1])), core((NOT, inner[2]))
			if inner[0] == ALL:
				return ANY, inner[1], core((NOT, inner[2]))
		return NOT, core(inner)
	if expr[0] == OR:
		assert len(expr) == 3
		return NOT, (AND, core((NOT, expr[1])), core((NOT, expr[2])))
	if expr[0] == AND:
		assert len(expr) == 3
		return expr[0], core(expr[1]), core(expr[2])
	if expr[0] == ALL:
		return NOT, (ANY, expr[1], core((NOT, expr[2])))
	if expr[0] == ANY:
		return expr[0], expr[1], core(expr[2])
	if expr[0] == SUB:
		return expr[0], core(expr[1]), core(expr[2])
	if expr[0] == DIS:
		assert len(expr) == 3
		return SUB, (AND, core(expr[1]), core(expr[2])), BOT
	assert False, f'bad expression {expr}'

def im(c, d):
	cxd = T.outer(c, d).view(-1)
	return T.cat((c, d, cxd))

class NeuralReasoner(nn.Module):
	def __init__(self, head=None, embs=None, emb_size=None, onto=None, hidden_size=None, hidden_count=1, seed=None):
		super().__init__()
		with T.random.fork_rng(enabled=seed is not None):
			if seed is not None: T.random.manual_seed(seed)

			self.head = head
			self.embs = embs

			if embs is not None:
				emb_size = self.embs.emb_size
			elif head is not None:
				emb_size = self.head.emb_size
			else:
				assert emb_size is not None

			if embs is None:
				assert onto is not None
				self.embs = EmbeddingLayer.from_onto(onto, emb_size=emb_size)

			if head is None:
				self.head = ReasonerHead(emb_size=emb_size, hidden_size=hidden_size, hidden_count=hidden_count)

			assert self.head.emb_size == self.embs.emb_size

	def encode(self, expr):
		return self.head.encode(core(expr), self.embs).detach().numpy()

	def check(self, axiom):
		return T.sigmoid(self.head.classify_batch([core(axiom)], [self.embs]))[0].item()

	def check_sub(self, c, d):
		return self.check((SUB, c, d))

class EmbeddingLayer(nn.Module):
	def __init__(self, *, emb_size, n_concepts, n_roles):
		super().__init__()
		self.n_concepts = n_concepts
		self.n_roles = n_roles
		self.emb_size = emb_size
		self.concepts = nn.Parameter(T.zeros((n_concepts, emb_size)))
		self.roles = nn.ModuleList([nn.Linear(emb_size, emb_size) for _ in range(n_roles)])
		nn.init.xavier_normal_(self.concepts)
		
	@classmethod
	def from_onto(cls, onto, *args, **kwargs):
		return cls(n_concepts=onto.n_concepts, n_roles=onto.n_roles, *args, **kwargs)

	@classmethod
	def from_ontos(cls, ontos, *args, **kwargs):
		return [cls.from_onto(onto, *args, **kwargs) for onto in ontos]

class ReasonerHead(nn.Module):
	def __init__(self, *, emb_size, hidden_size, hidden_count=1):
		super().__init__()
		self.hidden_size, self.emb_size = hidden_size, emb_size
		self.bot_concept = nn.Parameter(T.zeros((1, emb_size)))
		self.top_concept = nn.Parameter(T.zeros((1, emb_size)))
		self.not_nn = nn.Linear(emb_size, emb_size)
		self.and_nn = nn.Linear(2*emb_size + emb_size**2, emb_size)

		sub_nn = [nn.Linear(2*emb_size + emb_size**2, hidden_size)]
		for _ in range(hidden_count - 1):
			sub_nn.append(nn.ELU())
			sub_nn.append(nn.Linear(hidden_size, hidden_size))
		sub_nn.append(nn.ELU())
		sub_nn.append(nn.Linear(hidden_size, 1))
		self.sub_nn = nn.Sequential(*sub_nn)

		self.rvnn_act = lambda x: x
		#self.rvnn_act = T.tanh

		nn.init.xavier_normal_(self.bot_concept)
		nn.init.xavier_normal_(self.top_concept)
			
	def encode(self, axiom, embeddings):
		def rec(expr):
			if expr == TOP:
				return self.rvnn_act(self.top_concept[0])
			elif expr == BOT:
				return self.rvnn_act(self.bot_concept[0])
			elif isinstance(expr, int):
				return self.rvnn_act(embeddings.concepts[expr])
			elif expr[0] == AND:
				c = rec(expr[1])
				d = rec(expr[2])
				return self.rvnn_act(self.and_nn(im(c, d)))
			elif expr[0] == NOT:
				c = rec(expr[1])
				return self.rvnn_act(self.not_nn(c))
			elif expr[0] == ANY:
				c = rec(expr[2])
				r = embeddings.roles[expr[1]]
				return self.rvnn_act(r(c))
			elif expr[0] == SUB:
				c = rec(expr[1])
				d = rec(expr[2])
				return self.sub_nn(im(c, d))
			else:
				assert False, f'Unsupported expression {expr}. Did you convert it to core form?'
		return rec(axiom)
	
	def classify_batch(self, axioms, embeddings):
		return T.vstack([self.encode(axiom, emb) for axiom, emb in zip(axioms, embeddings)])
	
	def classify(self, axiom, emb):
		return self.classify_batch([axiom], [emb])[0].item()

def batch_stats(Y, y, **other):
	K = np.array(Y) > 0.5
	roc_auc = metrics.roc_auc_score(y, Y)
	pr_auc = metrics.average_precision_score(y, Y)
	acc = metrics.accuracy_score(y, K)
	f1 = metrics.f1_score(y, K)
	prec = metrics.precision_score(y, K, zero_division=0)
	recall = metrics.recall_score(y, K)
	return dict(acc=acc, f1=f1, prec=prec, recall=recall, roc_auc=roc_auc, pr_auc=pr_auc, **other)

def eval_batch(reasoner, encoders, X, y, onto_idx, indices=None, *, backward=False, detach=True):
	if indices is None: indices = list(range(len(X)))
	emb = [encoders[onto_idx[i]] for i in indices]
	X_ = [core(X[i]) for i in indices]
	y_ = T.tensor([float(y[i]) for i in indices]).unsqueeze(1)
	Y_ = reasoner.classify_batch(X_, emb)
	loss = F.binary_cross_entropy_with_logits(Y_, y_, reduction='mean')
	if backward:
		loss.backward()
	Y_ = T.sigmoid(Y_)
	if detach:
		loss = loss.item()
		y_ = y_.detach().numpy().reshape(-1)
		Y_ = Y_.detach().numpy().reshape(-1)
	return loss, list(y_), list(Y_)

def train(data_tr, data_vl, reasoner, encoders, *, epoch_count=15, batch_size=32, logger=None, validate=True,
		optimizer=T.optim.AdamW, lr_reasoner=0.0001, lr_encoder=0.0002, freeze_reasoner=False, run_name='train'):
	idx_tr, X_tr, y_tr = data_tr
	idx_vl, X_vl, y_vl = data_vl if data_vl is not None else data_tr
	if logger is None:
		logger = TrainingLogger(validate=validate, metrics=batch_stats)

	optimizers = []
	for encoder in encoders:
		optimizers.append(optimizer(encoder.parameters(), lr=lr_encoder))

	if freeze_reasoner:
		freeze(reasoner)
	else:
		optimizers.append(optimizer(reasoner.parameters(), lr=lr_reasoner))

	logger.begin_run(epoch_count=epoch_count, run=run_name)
	for epoch_idx in range(epoch_count + 1):
		# Training
		batches = minibatches(T.randperm(len(X_tr)), batch_size)
		logger.begin_epoch(batch_count=len(batches))
		for idxs in batches:
			for optim in optimizers:
				optim.zero_grad()
			loss, yb, Yb = eval_batch(reasoner, encoders, X_tr, y_tr, idx_tr, idxs, backward=epoch_idx > 0)
			for optim in optimizers:
				optim.step()
			logger.step(loss)

		# Validation
		if validate:
			with T.no_grad():
				val_loss, yb, Yb = eval_batch(reasoner, encoders, X_vl, y_vl, idx_vl)
				logger.step_validate(val_loss, yb, Yb, idx_vl)

		logger.end_epoch()

	if freeze_reasoner:
		unfreeze(reasoner)

	return logger
