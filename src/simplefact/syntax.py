TOP, BOT = -1, -2
SUB, EQV, DIS, NOT, AND, OR, ALL, ANY = range(8)
FORALL = ALL
EXISTS = ANY

#TO_SUBSCRIPT = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
OP_PRECEDENCE = {NOT: 50, ANY: 40, ALL: 40, AND: 30, OR: 20, SUB: 10, EQV: 10, DIS: 10}
OP_PRETTY = {NOT: '¬', AND: ' ⊓ ', OR: ' ⊔ ', SUB: ' ⊑ ', EQV: ' = ', DIS: ' != ', ANY: '∃', ALL: '∀'}
OP_BINARY = {AND, OR, SUB, EQV, DIS}
OP_QUANTIFIER = {ANY, ALL}
OP_UNARY = {NOT}

def expr_depth(x):
	if not isinstance(x, tuple): return 0
	return 1 + max(expr_depth(xx) for xx in x)

def to_pretty(expr, *, concept_names=None, role_names=None):
	"""
	Returns a pretty representation of a given expression.
	"""
	def rec(x, prec):
		if isinstance(x, tuple):
			head = x[0]
			head_prec = OP_PRECEDENCE[head]
			if head in OP_UNARY:
				result = OP_PRETTY[head] + rec(x[1], head_prec)
			elif head in OP_BINARY:
				result = OP_PRETTY[head].join(rec(xx, head_prec) for xx in x[1:])
			elif head in OP_QUANTIFIER:
				role = 'R' + str(x[1]) if role_names is None else role_names[x[1]]
				result = OP_PRETTY[head] + role + '.' + rec(x[2], head_prec)
			else:
				assert False, f'unknown operator {head}'
			if prec > head_prec:
				result = '(' + result + ')'
			return result
		elif x == TOP:
			return '⊤'
		elif x == BOT:
			return '⊥'
		elif isinstance(x, int):
			return 'C' + str(x) if concept_names is None else concept_names[x]
		else:
			assert False, f'bad expression {x}'
	return rec(expr, 0)

