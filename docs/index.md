# Welcome to Ploomber Example Docs!

<img src="https://images.viblo.asia/35852bff-d14e-457f-b562-00db7c0494cb.png" width="500" >


Ploomber is a Python library designed to streamline the development and deployment of data pipelines. It allows you to build pipelines using your favorite editor (Jupyter Notebook, VS Code, etc.) and deploy them to various platforms like Kubernetes, Airflow, and the cloud (Ploomber Cloud) without requiring significant code changes.

Here's a quick introduction to using Ploomber:

**1. Installation:**

Install Ploomber using pip:

```bash
pip install ploomber
```

**2. Defining your pipeline:**

Ploomber lets you define your pipeline using either:

* **Spec files (YAML):** These files define the pipeline structure explicitly, specifying tasks, dependencies, and parameters.
* **Jupyter Notebooks:** You can use specially formatted Jupyter Notebooks where code cells are treated as pipeline tasks.

**3. Running the pipeline:**

Once you define your pipeline, you can run it locally using the `ploomber run` command. This executes the tasks in the defined order.

**4. Deployment:**

Ploomber shines in its deployment capabilities. You can push your code (including notebooks or spec files) to a Git repository like GitHub. Ploomber automatically detects the pipeline definition and deploys it to your chosen platform with minimal adjustments.

**Additional features:**

* **Parametrization:** Easily manage pipeline configurations with parameters.
* **Report generation:** Generate reports automatically after pipeline execution.
* **Integration with various tools:** Works seamlessly with popular frameworks like TensorFlow, PyTorch, and scikit-learn.

**Getting started:**

Refer to the official Ploomber documentation [https://docs.ploomber.io/](https://docs.ploomber.io/) for detailed tutorials and examples to get you started with building and deploying your data pipelines.