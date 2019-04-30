import json

data_path = '../data/data.json'

def output_file(l, filepath):
	fout = open(filepath, 'w')
	for v in l:
		print(v, file = fout)
	fout.close()


def count():
	fin = open(data_path, 'r')
	data = []
	skills = []
	
	for line in fin:
		d = json.loads(line)
		data.append(d)
		skills += d['skills']
	city = list(set([d['city'] for d in data]))
	country = list(set([d['country'] for d in data]))
	skills = list(set(skills))
	request_wage = [d['log_requested_wage'] for d in data]
	real_wage = [d['log_realized_wage'] for d in data]

	print('skills:', len(skills))
	print('country:', len(country))

	output_file(skills, '../data/skills.txt')
	output_file(request_wage, '../data/request_wage.txt')
	output_file(real_wage, '../data/real_wage.txt')

count()

