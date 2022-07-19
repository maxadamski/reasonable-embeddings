from .syntax import *

class dotdict(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

class Literal:
	__slots__ = 'data', 'lang', 'dtype'

	def __init__(self, data: str, lang=None, dtype=None):
		self.data, self.lang, self.dtype = data, lang, dtype

	def __str__(self):
		return str(self.data)

	def __repr__(self):
		result = repr(self.data)
		if self.lang is not None: result = f'{result}@{self.lang}'
		if self.dtype is not None: result = f'{result}^^{self.dtype}'
		return result

class Onto:
	def __init__(self, *, tbox: set, n_concepts=0, n_roles=0, concepts=None, roles=None, iri=None, prefixes=None, annotations=None):
		self.iri = iri
		self.tbox = tbox
		self.n_concepts = n_concepts if concepts is None else len(concepts)
		self.n_roles = n_roles if roles is None else len(roles)
		self.prefixes = {} if prefixes is None else prefixes
		self.concepts = [] if concepts is None else concepts
		self.roles = [] if roles is None else roles
		self.annotations = [] if annotations is None else annotations
		self.did_change_names()

	def did_change_names(self):
		self.concept_by_name = dotdict({name: id for id, name in enumerate(self.concepts)})
		self.role_by_name = dotdict({name: id for id, name in enumerate(self.roles)})

	def use_annotations_as_names(self, prop='rdfs:label', lang='en'):
		concepts = {x: x for x in self.concepts}
		roles = {x: x for x in self.roles}
		for p, s, o in self.annotations:
			if p == prop and isinstance(o, Literal) and o.lang == lang:
				if s in concepts: concepts[s] = str(o)
				if s in roles: roles[s] = str(o)

		self.concepts = [concepts[c] for c in self.concepts]
		self.roles = [roles[r] for r in self.roles]
		self.did_change_names()

	def use_prefix(self, prefix=':'):
		self.concepts = [c.replace(prefix, '') for c in self.concepts]
		self.roles = [r.replace(prefix, '') for r in self.roles]
		self.did_change_names()

	def expand(self, iri):
		return expand_iri(iri, self.prefixes)

	def render(self, x):
		return to_pretty(x, concept_names=self.concepts, role_names=self.roles)

	def __repr__(self):
		return f'<Onto {self.n_concepts} concepts, {self.n_roles} roles, {len(self.tbox)} axioms>'

	__str__ = __repr__

def expand_iri(iri, prefixes):
	i = iri.find(':')
	if i == -1: return iri
	prefix, suffix = iri[:i], iri[i+1:]
	expanded_prefix = prefixes.get(prefix)
	assert expanded_prefix is not None, f'No expansion for prefix {prefix}:'
	return expanded_prefix + suffix

def load_ofn(path, **kwargs):
	with open(path, 'rb') as f:
		return parse_ofn(f.read(), **kwargs)

def load_owl(path, *, convert_script='scripts/ontoconvert', **kwargs):
	import subprocess
	text = subprocess.Popen([convert_script, path], stdout=subprocess.PIPE).communicate()[0]
	return parse_ofn(text, **kwargs)

def parse_ofn(data: bytes, *, raw=False, silent=False, **kwargs):
	exprs = parse_ofn_c(data)
	result = []
	for expr in exprs:
		head = expr[0]
		if head == 'Prefix':
			prefix, iri = expr[1].split('=', 1)
			iri = iri[1:-1]
			prefix = prefix[:-1]
			result.append(('Prefix', prefix, iri))
		else:
			result.append(expr)

	if not raw:
		result = pack_ofn(result, silent=silent)

	return result

def pack_ofn(data, *, silent=False):
	prefixes = {}
	onto = None
	for expr in data:
		head = expr[0]
		if head == 'Prefix':
			prefixes[expr[1]] = expr[2]
		elif head == 'Ontology':
			onto = expr

	if onto is None:
		raise Exception('No Ontology(...) found!')
			
	onto_iri = onto[1]
	concept_names = []
	declared_role_names = []
	role_names = set()
	labels = {}
	annot = []
	tbox = set()

	def parse_axiom(expr):
		if isinstance(expr, str):
			if expr == 'owl:Thing': return TOP
			if expr == 'owl:Nothing': return BOT
			return expr
			
		head = expr[0]
		if head == 'SubClassOf':
			return (SUB, parse_axiom(expr[1]), parse_axiom(expr[2]))
		elif head == 'DisjointClasses':
			return (DIS, *[parse_axiom(x) for x in expr[1:]])
		elif head == 'EquivalentClasses':
			return (EQV, *[parse_axiom(x) for x in expr[1:]])
		elif head == 'ObjectIntersectionOf':
			return (AND, *[parse_axiom(x) for x in expr[1:]])
		elif head == 'ObjectUnionOf':
			return (OR, *[parse_axiom(x) for x in expr[1:]])
		elif head == 'ObjectComplementOf':
			return (NOT, parse_axiom(expr[1]))
		elif head == 'ObjectSomeValuesFrom':
			result = (ANY, expr[1], parse_axiom(expr[2]))
			role_names.add(expr[1])
			return result
		elif head == 'ObjectAllValuesFrom':
			result = (ALL, expr[1], parse_axiom(expr[2]))
			role_names.add(expr[1])
			return result
		else:
			assert False

	for expr in onto[2:]:
		head = expr[0]
		if head == 'Declaration':
			decl = expr[1]
			typ = decl[0]
			if typ == 'Class':
				name = decl[1]
				if name in {'owl:Thing', 'owl:Nothing'}: continue
				concept_names.append(name)
			elif typ == 'ObjectProperty':
				declared_role_names.append(decl[1])
		
		elif head in {'SubClassOf', 'EquivalentClasses', 'DisjointClasses'}:
			try:
				axiom = parse_axiom(expr)
				tbox.add(axiom)
			except:
				if not silent: print('Unsupported class expression', expr)

		elif head == 'AnnotationAssertion':
			annot.append((expr[1], expr[2], expr[3]))

		elif head == 'ObjectPropertyDomain':
			role_names.add(expr[1])
			tbox.add((SUB, (ANY, expr[1], TOP), parse_axiom(expr[2]) ))

		elif head == 'ObjectPropertyRange':
			role_names.add(expr[1])
			tbox.add((SUB, TOP, (ALL, expr[1], parse_axiom(expr[2]) )))
					
	if not silent:
		for role in declared_role_names:
			if role not in role_names:
				print('Dropping unused role', role)

	role_names = [name for name in role_names]
	concept_name_to_idx = {name: i for i, name in enumerate(concept_names)}
	role_name_to_idx = {name: i for i, name in enumerate(role_names)}

	def rename(expr):
		if isinstance(expr, str):
			return concept_name_to_idx[expr]
		if not isinstance(expr, tuple):
			return expr
		head = expr[0]
		if head in {ALL, ANY}:
			return (head, role_name_to_idx[expr[1]], rename(expr[2]))
		else:
			return (head, *[rename(x) for x in expr[1:]])

	tbox = {rename(x) for x in tbox}

	return Onto(tbox=tbox, concepts=concept_names, roles=role_names, iri=onto_iri, prefixes=prefixes, annotations=annot)

cdef struct Parser:
	int pos
	int total
	char *data

cdef char peek_unsafe(Parser *p):
	return p.data[p.pos]

cdef char peek(Parser *p):
	if p.pos < p.total: return peek_unsafe(p)
	return b'\0'

cdef char until(Parser *p, char c):
	return p.pos < p.total and peek_unsafe(p) != c

cdef str parse_id(Parser *p):
	start = p.pos
	cdef char curr
	while p.pos < p.total:
		curr = peek_unsafe(p)
		if curr == b' ' or curr == b'(' or curr == b')':
			break
		p.pos += 1
	return p.data[start:p.pos].decode('UTF-8')

cdef str parse_iri(Parser *p):
	p.pos += 1
	start = p.pos
	while until(p, b'>'):
		p.pos += 1
	text = p.data[start:p.pos].decode('UTF-8')
	p.pos += 1
	return text

cdef object parse_str(Parser *p):
	p.pos += 1
	start = p.pos
	while until(p, b'"'):
		p.pos += 1
		if peek(p) == b'\\':
			p.pos += 2
	text = p.data[start:p.pos].decode('UTF-8')
	p.pos += 1

	lang = None
	dtype = None
	parse_ws(p)
	if peek(p) == b'@':
		p.pos += 1
		lang = parse_id(p)
	if peek(p) == b'^':
		p.pos += 2
		parse_ws(p)
		dtype = parse_id(p)
	return Literal(text, lang=lang, dtype=dtype)

cdef void parse_comment(Parser *p):
	while until(p, b'\n'):
		p.pos += 1
	p.pos += 1

cdef void parse_ws(Parser *p):
	cdef char curr
	while p.pos < p.total:
		curr = peek_unsafe(p)
		if curr != b' ' and curr != b'\t' and curr != b'\n':
			break
		p.pos += 1

cdef void parse_whitespace(Parser *p):
	parse_ws(p)
	while peek(p) == b'#':
		parse_comment(p)
		parse_ws(p)

cdef object parse_expr(Parser *p):
	parse_whitespace(p)
	if p.pos < p.total:
		curr = peek_unsafe(p)
		if curr == b'"':
			return parse_str(p)
		if curr == b'<':
			return parse_iri(p)

	head = parse_id(p)
	if p.pos < p.total and peek_unsafe(p) != b'(':
		return head

	p.pos += 1
	parse_whitespace(p)
	expr = [head]
	while until(p, b')'):
		arg = parse_expr(p)
		expr.append(arg)
		parse_whitespace(p)
	p.pos += 1
	return tuple(expr)

cdef list parse_ofn_c(data: bytes):
	cdef Parser p
	p.pos = 0
	p.data = data
	p.total = len(data)
	result = []
	while p.pos < p.total:
		result.append(parse_expr(&p))
	return result

__all__ = 'Onto', 'Literal', 'expand_iri', 'load_ofn', 'load_owl', 'parse_ofn', 'pack_ofn'

