import json, urllib.request, urllib.error

PROJECT_ID = 'prj_zfotYsBhNBXYOs8XgEgQs1dZIUag'
ORG_ID = 'team_5nwk1lnyeT488lDmbz10yu0s'

with open('/Users/joergpeetz/Library/Application Support/com.vercel.cli/auth.json') as f:
    TOKEN = json.load(f)['token']

url = f'https://api.vercel.com/v9/projects/{PROJECT_ID}?teamId={ORG_ID}'
# Set outputDirectory to frontend/.vercel/output (where adapter-vercel puts it)
body = json.dumps({
    'outputDirectory': 'frontend/.vercel/output',
    'buildCommand': 'cd frontend && npm run build',
    'installCommand': 'cd frontend && npm install',
    'nodeVersion': '24.x'
}).encode()
req = urllib.request.Request(
    url, data=body,
    headers={'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'},
    method='PATCH'
)
try:
    resp = urllib.request.urlopen(req)
    d = json.loads(resp.read())
    print(f'outputDirectory: {d.get("outputDirectory")}')
    print(f'buildCommand: {d.get("buildCommand")}')
except urllib.error.HTTPError as e:
    print(f'HTTP {e.code}: {e.read().decode()[:500]}')