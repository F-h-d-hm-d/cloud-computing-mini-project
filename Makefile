.PHONY: up down status test

up:
k3d cluster start cloud-assignment

down:
k3d cluster stop cloud-assignment

status:
kubectl get pods -A

test:
python -m unittest discover -s tests -p "test_*.py" -v
