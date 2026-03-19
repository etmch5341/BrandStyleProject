export $(cat .env | xargs)
~/brand-gen-env/bin/uvicorn main:app --host 0.0.0.0 --port 8000
