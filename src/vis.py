import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.decomposition import PCA
from simplefact.syntax import *

pca = None

sns.set_context('talk')
sns.set_style('white')
plt.rc('figure', figsize=(15, 8), dpi=200)

def plot_train_history(train_history, valid_history, save=False):
	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
	#plt.xticks(valid_history.epoch, valid_history.epoch, rotation=90)
	sns.lineplot(x='epoch', y='batch_loss', data=train_history, ci='sd', label='Training loss', ax=ax1)
	ax1.plot(valid_history.val_loss, label='Validation loss')
	ax1.legend(loc='upper right')
	ax1.set_ylabel('BCE loss')
	ax1.grid()

	ax2.plot(valid_history.roc_auc, label='Validation AUC ROC')
	ax2.set_xlim(0, valid_history.epoch.max())
	ax2.set_ylim(0.4, 1)
	ax2.set_ylabel('AUC ROC')
	ax2.set_xlabel('Epoch')
	ax2.legend(loc='lower right')
	ax2.grid()

	if save: plt.savefig(save)
	plt.show()

def plot_test_history(test_history, test_history_by_onto, save=False):
	fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
	#plt.xticks(range(epochs + 1), rotation=90)
	sns.lineplot(x='epoch', y='val_loss', hue='run', ci='sd', data=test_history, ax=ax1)
	ax1.set_ylabel('BCE loss')
	ax1.legend(loc='upper right')
	ax1.grid()

	sns.lineplot(x='epoch', y='roc_auc', hue='run', ci=None, data=test_history_by_onto, ax=ax2)
	#sns.boxplot(x='epoch', y='roc_auc', hue='run', data=test_history_by_onto, ax=ax2)
	sns.lineplot(x='epoch', y='roc_auc', hue='run', units='onto', estimator=None, alpha=0.4, ls='--', lw=1, legend=False, data=test_history_by_onto, ax=ax2)
	ax2.set_ylabel('AUC ROC')
	ax2.set_xlabel('Test epoch')
	ax2.legend(loc='lower right')
	ax2.set_xlim(0, test_history.epoch.max())
	ax2.set_ylim(0.4, 1)
	ax2.grid()

	if save: plt.savefig(save)
	plt.show()

def test_metrics(y, Y, thresh=0.5):
	return dict(
		acc=metrics.accuracy_score(y, Y > thresh),
		prec = metrics.precision_score(y, Y > thresh),
		recall = metrics.recall_score(y, Y > thresh),
		f1=metrics.f1_score(y, Y > thresh),
		auc_roc=metrics.roc_auc_score(y, Y),
		auc_pr=metrics.average_precision_score(y, Y),
	)

def report(ontos, y, Y, idx, thresh=0.5, save=False):
	fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(24, 8))
	Y = np.array(Y).reshape(-1)
	y = np.array(y)
	
	ms = []
	for onto_idx in range(len(ontos)):
		mask = np.array(idx) == onto_idx
		y_onto, Y_onto = y[mask], Y[mask]
		fpr, tpr, _ = metrics.roc_curve(y_onto, Y_onto)
		prec, recall, _ = metrics.precision_recall_curve(y_onto, Y_onto)
		ax1.plot(fpr, tpr, alpha=0.3, lw=1, color='tab:blue')
		ax2.plot(recall, prec, alpha=0.3, lw=1, color='tab:blue')
		ms.append(test_metrics(y_onto, Y_onto, thresh))
		
	fpr, tpr, _ = metrics.roc_curve(y, Y)
	prec, recall, _ = metrics.precision_recall_curve(y, Y)
	ax1.plot(fpr, tpr, lw=3, color='tab:blue')
	ax2.plot(recall, prec, lw=3, color='tab:blue')
	ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
	ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')

	for ax in (ax1, ax2):
		ax.set_xlim(-0.02, 1.02)
		ax.set_ylim(-0.02, 1.02)
		ax.set_aspect('equal')
		ax.grid()

	if save: plt.savefig(save.replace('%', 'curves') + '.png')
		
	disp = metrics.ConfusionMatrixDisplay.from_predictions(y, Y > thresh, normalize='all', cmap='cividis', values_format='.2%', colorbar=False, ax=ax3)
	#confusion = metrics.confusion_matrix(y, Y > thresh)
	#sns.heatmap(confusion/np.sum(confusion), annot=True, fmt='.2%', cmap='cividis', cbar=False, ax=ax3)

	#if save: plt.savefig(save.replace('%', 'matrix') + '.png')
	
	df = pd.DataFrame(ms).agg(['mean', 'std']).T
	df['micro'] = pd.Series(test_metrics(y, Y, thresh))
	if save: df.round(4).to_csv(save.replace('%', 'stats') + '.csv')

	return df

def vis_pizza(onto, fact, neur, seed=2022, umap=True, n_neighbors=50, min_dist=0.2, reload_pca=False, ax=None):
	global pca 

	if ax is None:
		fig, ax = plt.subplots()

	if pca is None or reload_pca:
		if umap:
			from umap import UMAP
			pca = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
		else:
			pca = PCA(n_components=2, random_state=seed)

	C = onto.concept_by_name
	exclude = {}
	manual_exprs = [C[x] for x in C if x not in exclude]
	manual_exprs += [TOP, BOT]

	exprs, texts, colors, markers = [], [], [], []
	for expr in manual_exprs:
		exprs.append(neur.encode(expr))
		texts.append(onto.render(expr))

		if fact.check_eqv(expr, BOT):
			m = 's'
		elif fact.check_sub(expr, C.Pizza):
			m = 'o'
		elif fact.check_sub(expr, C.PizzaTopping):
			m = 'd'
		else:
			m = 's'

		c = 'gray'
		if isinstance(expr, tuple) or expr in {TOP, BOT}:
			c = 'cyan'
			m = 's'
		elif not fact.check_sub(expr, BOT):
			if fact.check_sub(expr, C.VegetarianPizza): c = 'lime'
			if fact.check_sub(expr, C.NonVegetarianPizza): c = 'brown'

			if fact.check_sub(expr, C.VegetarianTopping): c = 'lime'
			if fact.check_sub(expr, C.VegetableTopping): c = 'green'
			if fact.check_sub(expr, C.MeatTopping): c = 'brown'
			if fact.check_sub(expr, C.SeafoodTopping): c = 'pink'
			if fact.check_sub(expr, C.Spiciness): c = 'red'

		colors.append(c)
		markers.append(m)

	exprs = np.array(exprs)
	if exprs.shape[1] == 2:
		xs, ys = exprs.T
	else:
		xs, ys = pca.fit_transform(exprs).T

	for x, y, text, color, marker, expr in zip(xs, ys, texts, colors, markers, manual_exprs):
		ax.scatter(x, y, marker=marker, color=color)
		if not (isinstance(expr, tuple) or fact.check_sub(expr, BOT)):
			if fact.check_sub(expr, (OR, C.CheeseTopping, C.CheesyPizza)):
				ax.scatter(x, y, marker=marker, color='yellow', s=30)
			if fact.check_sub(expr, C.PepperTopping):
				ax.scatter(x, y, marker=marker, color='orange', s=30)
			if fact.check_sub(expr, (OR, C.SpicyTopping, C.SpicyPizza)):
				ax.scatter(x, y, marker='x', color='red', s=50)
		ax.text(x, y, text.replace('Topping', 'T'), fontsize=10)

