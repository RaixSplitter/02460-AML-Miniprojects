


flow_train:
	python miniproject1/flow_mnist.py train --model model_flow_mnist --samples samples_flow_mnist --epochs 20 --batch-size 64

flow_sample:
	python miniproject1/flow_mnist.py sample --model model_flow_mnist --samples samples_flow_mnist --epochs 5 --batch-size 64

flow_train_cb:
	python miniproject1/flow.py train --data cb --model model_flow_cb2.pt --samples samples_flow_cb2.png --epochs 20

flow_sample_cb:
	python miniproject1/flow.py sample --data cb --model model_flow_cb2.pt --samples samples_flow_cb2.png


ddpm_train:
	python miniproject1/ddpm_mnist.py train --model models/model_ddpm_mnist.pt --samples samples_output/samples_ddpm_mnist --epochs 50 --batch-size 64

ddpm_sample:
	python miniproject1/ddpm_mnist.py sample --model models/model_ddpm_mnist.pt --samples samples_output/samples_ddpm_mnist --epochs 50 --batch-size 64


run_geo:
	python miniproject2/ensemble_vae.py geodesics