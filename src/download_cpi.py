import requests
import json

headers = {'Content-type': 'application/json'}
data = json.dumps({"seriesid": ['CUSR0000SA0'],
                  "startyear":"2008",
                  "endyear":"2017"})
p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/',
                  data=data,
                  headers=headers)
json_data = json.loads(p.text)
df = pd.DataFrame(json.loads(p.text)['Results']['series'][0]['data'])
df = df.drop(['footnotes'], axis=1)
df['month'] = df['period'].str.replace('M', '').astype(int)
df['year'] = df['year'].astype(int)
df['value'] = df['value'].astype(float)
df = df.drop(['period'], axis=1)
df = df.drop(['periodName'], axis=1)
df['date']= df.apply(lambda x:datetime.strptime("{0} {1}".format(int(x['year']),
                                                int(x['month'])),
                                                "%Y %m"),axis=1)
df = df.drop(['year', 'month'], axis=1)
