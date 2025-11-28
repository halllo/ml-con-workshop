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

Instead I switched to a github devspace with 32GB.

I install dependencies one by one and wait for all pods of each command to get ready before I run the next command.

```bash
./scripts/install-kubeflow.sh
./scripts/install-kserve.sh
kubectl apply -f kserve/rbac-fix.yaml
```

I went into the solution folder directly and built the pipeline:

```bash
python3 recommendation_pipeline_solution.py --output pipeline_with_deploy.yaml
```

Then I upload it to kubeflow and start a run.

It fails at downloading the movie dataset:

```log
ime="2025-11-28T15:18:45.446Z" level=info msg="capturing logs" argo=true
I1128 15:18:45.556450      27 main.go:61] Setting log level to: '1'
I1128 15:18:45.561418      27 cache.go:77] Connecting to cache endpoint 10.96.119.219:8887
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: 
https://pip.pypa.io/warnings/venv
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: 
https://pip.pypa.io/warnings/venv
[notice] A new release of pip is available: 24.0 -> 25.3
[notice] To update, run: pip install --upgrade pip
[KFP Executor 2025-11-28 15:19:04,910 INFO]: Looking for component `prepare_data` in --component_module_path `/tmp/tmp.skyAgR3HU3/ephemeral_component.py`
[KFP Executor 2025-11-28 15:19:04,910 INFO]: Loading KFP component "prepare_data" from /tmp/tmp.skyAgR3HU3/ephemeral_component.py (directory "/tmp/tmp.skyAgR3HU3" and module name "ephemeral_component")
[KFP Executor 2025-11-28 15:19:04,911 INFO]: Got executor_input:
{
    "inputs": {
        "parameterValues": {
            "dataset_size": "100k",
            "random_state": 42,
            "test_ratio": 0.2
        }
    },
    "outputs": {
        "artifacts": {
            "executor-logs": {
                "artifacts": [
                    {
                        "type": {
                            "schemaTitle": "system.Artifact"
                        },
                        "uri": "minio://mlpipeline/v2/artifacts/movie-recommendation-pipeline/bb837ea5-145f-46ef-8551-3e67af078630/prepare-data/ace5ed54-7dd2-474e-84ed-681d2beb05c1/executor-logs"
                    }
                ]
            },
            "movies_metadata": {
                "artifacts": [
                    {
                        "type": {
                            "schemaTitle": "system.Dataset",
                            "schemaVersion": "0.0.1"
                        },
                        "uri": "minio://mlpipeline/v2/artifacts/movie-recommendation-pipeline/bb837ea5-145f-46ef-8551-3e67af078630/prepare-data/ace5ed54-7dd2-474e-84ed-681d2beb05c1/movies_metadata"
                    }
                ]
            },
            "test_data": {
                "artifacts": [
                    {
                        "type": {
                            "schemaTitle": "system.Dataset",
                            "schemaVersion": "0.0.1"
                        },
                        "uri": "minio://mlpipeline/v2/artifacts/movie-recommendation-pipeline/bb837ea5-145f-46ef-8551-3e67af078630/prepare-data/ace5ed54-7dd2-474e-84ed-681d2beb05c1/test_data"
                    }
                ]
            },
            "train_data": {
                "artifacts": [
                    {
                        "type": {
                            "schemaTitle": "system.Dataset",
                            "schemaVersion": "0.0.1"
                        },
                        "uri": "minio://mlpipeline/v2/artifacts/movie-recommendation-pipeline/bb837ea5-145f-46ef-8551-3e67af078630/prepare-data/ace5ed54-7dd2-474e-84ed-681d2beb05c1/train_data"
                    }
                ]
            }
        },
        "outputFile": "/tmp/kfp_outputs/output_metadata.json"
    }
}
[KFP Executor 2025-11-28 15:19:05,732 INFO]: Downloading MovieLens dataset from 
https://files.grouplens.org/datasets/movielens/ml-100k.zip
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/urllib/request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/usr/local/lib/python3.11/http/client.py", line 1303, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/usr/local/lib/python3.11/http/client.py", line 1349, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/usr/local/lib/python3.11/http/client.py", line 1298, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/usr/local/lib/python3.11/http/client.py", line 1058, in _send_output
    self.send(msg)
  File "/usr/local/lib/python3.11/http/client.py", line 996, in send
    self.connect()
  File "/usr/local/lib/python3.11/http/client.py", line 1468, in connect
    super().connect()
  File "/usr/local/lib/python3.11/http/client.py", line 962, in connect
    self.sock = self._create_connection(
                ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/socket.py", line 863, in create_connection
    raise exceptions[0]
  File "/usr/local/lib/python3.11/socket.py", line 848, in create_connection
    sock.connect(sa)
TimeoutError: [Errno 110] Connection timed out
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/usr/local/lib/python3.11/site-packages/kfp/dsl/executor_main.py", line 109, in <module>
    executor_main()
  File "/usr/local/lib/python3.11/site-packages/kfp/dsl/executor_main.py", line 101, in executor_main
    output_file = executor.execute()
                  ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/site-packages/kfp/dsl/executor.py", line 377, in execute
    result = self.func(**func_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/tmp.skyAgR3HU3/ephemeral_component.py", line 43, in prepare_data
    urllib.request.urlretrieve(dataset_url, zip_path)
  File "/usr/local/lib/python3.11/urllib/request.py", line 241, in urlretrieve
    with contextlib.closing(urlopen(url, data)) as fp:
                            ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/urllib/request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/urllib/request.py", line 519, in open
    response = self._open(req, data)
               ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/urllib/request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/urllib/request.py", line 496, in _call_chain
    result = func(*args)
             ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/urllib/request.py", line 1391, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/urllib/request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 110] Connection timed out>
I1128 15:21:22.502999      27 launcher_v2.go:181] publish success.
I1128 15:21:22.529018      27 client.go:738] Attempting to update DAG state
F1128 15:21:22.552240      27 main.go:53] failed to execute component: exit status 1
time="2025-11-28T15:21:23.490Z" level=info msg="sub-process exited" argo=true error="<nil>"
Error: exit status 1
```

I check if the url can actually be reached from within the cluster:

```bash
kubectl run wget-test --image=busybox --rm -it --restart=Never -- wget --timeout=10 --tries=1 https://files.grouplens.org/datasets/movielens/ml-100k.zip
```

It fails after a few seconds:

```log
All commands and output from this session will be recorded in container logs, including credentials and sensitive information passed through the command prompt.
If you don't see a command prompt, try pressing enter.
wget: download timed out
pod "wget-test" deleted from default namespace
pod default/wget-test terminated (Error)
```

I check if the url can be reached outside the cluster but inside GH codespaces:

```bash
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
```

It also times out after 2 minutes, same behavior as kubeflow:

```log
-2025-11-28 15:43:28--  https://files.grouplens.org/datasets/movielens/ml-100k.zip
Resolving files.grouplens.org (files.grouplens.org)... 128.101.96.204
Connecting to files.grouplens.org (files.grouplens.org)|128.101.96.204|:443... failed: Connection timed out.
Retrying.

--2025-11-28 15:45:45--  (try: 2)  https://files.grouplens.org/datasets/movielens/ml-100k.zip
Connecting to files.grouplens.org (files.grouplens.org)|128.101.96.204|:443... failed: Connection timed out.
Retrying.
```

Todo: How to allow this network request from GH CodeSpaces?
