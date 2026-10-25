"""Run the core upgraded X-ray experiment suite sequentially."""
import subprocess,sys
commands=[
 [sys.executable,'centralized_baseline.py'],
 [sys.executable,'run_federated_simulation.py','--rounds','5','--clients','3','--iid'],
 [sys.executable,'run_federated_simulation.py','--rounds','5','--clients','3','--non-iid'],
 [sys.executable,'run_federated_simulation.py','--rounds','5','--clients','3','--iid','--dp-noise','0.1','--label','federated_dp_01'],
 [sys.executable,'run_federated_simulation.py','--rounds','5','--clients','3','--iid','--compression','--label','federated_compression'],
 [sys.executable,'run_federated_simulation.py','--rounds','5','--clients','3','--iid','--dp-noise','0.1','--compression','--label','federated_dp_compression'],
]
for c in commands: subprocess.run(c,check=True)
