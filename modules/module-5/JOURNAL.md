# kubeflow

Installation

```bash
./scripts/install-kubeflow.sh
```

The default kind worker with 2GB did not suffice.

I add a second wroker:

```yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: mlops-workshop
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"    
  extraPortMappings:
  - containerPort: 80
    hostPort: 1080
    protocol: TCP
  - containerPort: 443
    hostPort: 10443
    protocol: TCP
- role: worker
- role: worker  # Add second worker
```

After this more pods start and go into running state.

But I still experience problems with not being ready.

Lets check how much the podman machine has:

```bash
podman machine inspect
```

It has only 2GB.

```json
[
     {
          "ConfigDir": {
               "Path": "/Users/manuelnaujoks/.config/containers/podman/machine/applehv"
          },
          "ConnectionInfo": {
               "PodmanSocket": {
                    "Path": "/var/folders/kh/.../T/podman/podman-machine-default-api.sock"
               },
               "PodmanPipe": null
          },
          "Created": "2024-04-22T09:11:59.799177+02:00",
          "LastUp": "2025-11-25T14:13:28.656754+01:00",
          "Name": "podman-machine-default",
          "Resources": {
               "CPUs": 4,
               "DiskSize": 100,
               "Memory": 2048,
               "USBs": []
          },
          "SSHConfig": {
               "IdentityPath": "/Users/manuelnaujoks/.local/share/containers/podman/machine/machine",
               "Port": 49683,
               "RemoteUsername": "core"
          },
          "State": "running",
          "UserModeNetworking": true,
          "Rootful": false,
          "Rosetta": true
     }
]
```

I would need at least 8GB of RAM.

To change it I need to recreate the podman machine and the kind cluster:

```bash
# Stop the Podman machine
podman machine stop

# Recreate with more resources
podman machine rm
podman machine init --memory 8192 --cpus 4 --disk-size 50  # 8GB RAM, 4 CPUs, 50GB disk
podman machine start

# Then recreate your kind cluster
kind delete cluster --name mlops-workshop
kind create cluster --config=<your-config.yaml>
```
