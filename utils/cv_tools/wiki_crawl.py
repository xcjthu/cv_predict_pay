import urllib3
import os

http = urllib3.PoolManager()

from selenium import webdriver

driver_path = '/Users/xcj/Desktop/law_ai/chromedriver'
browser = webdriver.Chrome(executable_path = driver_path)

url = 'https://en.wikipedia.org/wiki/%s'

fin = open('/Users/xcj/Desktop/law_ai/cv_pay/skills.txt', 'r')
'''
headers = {
	':authority': 'en.wikipedia.org',
	':method': 'GET',
	':path': '/wiki/Molecule_editor',
	':scheme': 'https',
	'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
	'accept-encoding': 'gzip, deflate, br',
	'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8',
	'cache-control': 'max-age=0',
	'cookie': 'WMF-Last-Access-Global=21-Jun-2019; GeoIP=JP:27:Osaka:34.67:135.49:v4; WMF-Last-Access=21-Jun-2019; mwPhp7Seed=f3b',
	'if-modified-since': 'Tue, 04 Jun 2019 19:38:19 GMT',
	'upgrade-insecure-requests': '1',
	'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}
'''

for s in fin:
	temp_url = url % s.strip().replace('-', '_')
	print(temp_url)
	# data = http.request('GET', temp_url)# , headers = headers)
	try:
		if os.path.exists('/Users/xcj/Desktop/law_ai/cv_pay/wiki/%s' % (s.strip() + '.html')):
			continue
		browser.get(temp_url)
		html = browser.page_source # data.data.decode('utf8')

		fout = open('/Users/xcj/Desktop/law_ai/cv_pay/wiki/%s' % (s.strip() + '.html'), 'w')
		print(html, file = fout)
		print(s)
	except:
		pass
