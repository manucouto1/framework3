---
icon: material/home
---

# Framework3: A Machine Learning Model Development Platform

> **⚠️ Warning**: This project is currently under development and has not been thoroughly tested yet. Use with caution in production environments.




<div style="background: black; color: white; font-family: monospace; font-size: 15px;  ">
    <pre style="font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><code style="font-family:inherit"><span style="color: #800080; text-decoration-color: #800080; font-weight: bold; ">F3Pipeline</span><span style="font-weight: bold">(</span>
    <span style="color: #808000; text-decoration-color: #808000">plugins</span>=<span style="font-weight: bold">[</span>
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">MapReduceFeatureExtractorPipeline</span><span style="font-weight: bold">(</span>
            <span style="color: #808000; text-decoration-color: #808000">filters</span>=<span style="font-weight: bold">[</span>
                <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">F3Pipeline</span><span style="font-weight: bold">(</span>
                    <span style="color: #808000; text-decoration-color: #808000">plugins</span>=<span style="font-weight: bold">[</span>
                        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">StandardScalerPlugin</span><span style="font-weight: bold">()</span>,
                        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Cached</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">filter</span>=<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">PCAPlugin</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">n_components</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span><span style="font-weight: bold">)</span>, <span style="color: #808000; text-decoration-color: #808000">cache_data</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #808000; text-decoration-color: #808000">cache_filter</span>=<span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>, <span style="color: #808000; text-decoration-color: #808000">overwrite</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>, <span style="color: #808000; text-decoration-color: #808000">storage</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="font-weight: bold">)</span>,
                        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ClassifierSVMPlugin</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">C</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.0</span>, <span style="color: #808000; text-decoration-color: #808000">kernel</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;linear&#x27;</span>, <span style="color: #808000; text-decoration-color: #808000">gamma</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;scale&#x27;</span><span style="font-weight: bold">)</span>
                    <span style="font-weight: bold">]</span>,
                    <span style="color: #808000; text-decoration-color: #808000">metrics</span>=<span style="font-weight: bold">[</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">F1</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">average</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;weighted&#x27;</span><span style="font-weight: bold">)</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Precission</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">average</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;weighted&#x27;</span><span style="font-weight: bold">)</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Recall</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">average</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;weighted&#x27;</span><span style="font-weight: bold">)]</span>,
                    <span style="color: #808000; text-decoration-color: #808000">overwrite</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
                    <span style="color: #808000; text-decoration-color: #808000">store</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
                    <span style="color: #808000; text-decoration-color: #808000">log</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
                <span style="font-weight: bold">)</span>,
                <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">F3Pipeline</span><span style="font-weight: bold">(</span>
                    <span style="color: #808000; text-decoration-color: #808000">plugins</span>=<span style="font-weight: bold">[</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">StandardScalerPlugin</span><span style="font-weight: bold">()</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">PCAPlugin</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">n_components</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span><span style="font-weight: bold">)</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ClassifierSVMPlugin</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">C</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.0</span>, <span style="color: #808000; text-decoration-color: #808000">kernel</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;rbf&#x27;</span>, <span style="color: #808000; text-decoration-color: #808000">gamma</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;scale&#x27;</span><span style="font-weight: bold">)]</span>,
                    <span style="color: #808000; text-decoration-color: #808000">metrics</span>=<span style="font-weight: bold">[]</span>,
                    <span style="color: #808000; text-decoration-color: #808000">overwrite</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
                    <span style="color: #808000; text-decoration-color: #808000">store</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
                    <span style="color: #808000; text-decoration-color: #808000">log</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
                <span style="font-weight: bold">)</span>,
                <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">F3Pipeline</span><span style="font-weight: bold">(</span>
                    <span style="color: #808000; text-decoration-color: #808000">plugins</span>=<span style="font-weight: bold">[</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">StandardScalerPlugin</span><span style="font-weight: bold">()</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">PCAPlugin</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">n_components</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">)</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">ClassifierSVMPlugin</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">C</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1.0</span>, <span style="color: #808000; text-decoration-color: #808000">kernel</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;linear&#x27;</span>, <span style="color: #808000; text-decoration-color: #808000">gamma</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;scale&#x27;</span><span style="font-weight: bold">)]</span>,
                    <span style="color: #808000; text-decoration-color: #808000">metrics</span>=<span style="font-weight: bold">[]</span>,
                    <span style="color: #808000; text-decoration-color: #808000">overwrite</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
                    <span style="color: #808000; text-decoration-color: #808000">store</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
                    <span style="color: #808000; text-decoration-color: #808000">log</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
                <span style="font-weight: bold">)</span>
            <span style="font-weight: bold">]</span>,
            <span style="color: #808000; text-decoration-color: #808000">numSlices</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>
        <span style="font-weight: bold">)</span>,
        <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">KnnFilter</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">n_neighbors</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">weights</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;uniform&#x27;</span>, <span style="color: #808000; text-decoration-color: #808000">algorithm</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;auto&#x27;</span>, <span style="color: #808000; text-decoration-color: #808000">leaf_size</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">30</span>, <span style="color: #808000; text-decoration-color: #808000">p</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>, <span style="color: #808000; text-decoration-color: #808000">metric</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;minkowski&#x27;</span>, <span style="color: #808000; text-decoration-color: #808000">metric_params</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>, <span style="color: #808000; text-decoration-color: #808000">n_jobs</span>=<span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="font-weight: bold">)</span>
    <span style="font-weight: bold">]</span>,
    <span style="color: #808000; text-decoration-color: #808000">metrics</span>=<span style="font-weight: bold">[</span><span style="color: #800080; text-decoration-color: #800080; font-weight: bold">F1</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">average</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;weighted&#x27;</span><span style="font-weight: bold">)</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Precission</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">average</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;weighted&#x27;</span><span style="font-weight: bold">)</span>, <span style="color: #800080; text-decoration-color: #800080; font-weight: bold">Recall</span><span style="font-weight: bold">(</span><span style="color: #808000; text-decoration-color: #808000">average</span>=<span style="color: #008000; text-decoration-color: #008000">&#x27;weighted&#x27;</span><span style="font-weight: bold">)]</span>,
    <span style="color: #808000; text-decoration-color: #808000">overwrite</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
    <span style="color: #808000; text-decoration-color: #808000">store</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
    <span style="color: #808000; text-decoration-color: #808000">log</span>=<span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
<span style="font-weight: bold">)</span>
</div>

## Purpose

Framework3 is an innovative platform designed to simplify and accelerate the development of machine learning models. Our main goal is to provide data scientists and machine learning engineers with a flexible and powerful tool that allows them to create, experiment with, and deploy models efficiently and in a structured manner.

## Philosophy

Our philosophy is based on three fundamental principles:

1. **Flexibility**: We believe that each machine learning project is unique. That's why Framework3 is designed to be highly adaptable, allowing users to customize every aspect of the model development process.

2. **Modularity**: We have adopted a modular approach that allows users to combine different components easily. This facilitates experimentation and code reuse.

3. **Transparency**: We value clarity and understanding at every step of the machine learning process. Framework3 is designed to provide visibility at all stages, from data preparation to model evaluation.

## Architecture

Framework3 is built on a robust and scalable architecture:

### Main Components

1. **Base**: Includes the base classes and types that serve as the foundation for the entire framework.

2. **Container**: Provides structures to organize and manage the different components of the project.

3. **Plugins**: An extensible system that allows adding new functionalities:
   - Pipelines: For creating complex workflows.
   - Filters: For data transformation and processing.
   - Metrics: For model evaluation.
   - Storage: For efficient management of data and models.

4. **MapReduce**: Facilitates distributed processing to handle large volumes of data.

5. **Utilities**: A set of auxiliary tools for common tasks in model development.
