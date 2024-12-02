---
icon: material/home
---

# Framework3: Una Plataforma de Desarrollo de Modelos de Machine Learning

> **⚠️ Advertencia**: Este proyecto está actualmente en desarrollo y aún no ha sido probado exhaustivamente. Úselo con precaución en entornos de producción.

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


## Propósito

Framework3 es una plataforma innovadora diseñada para simplificar y acelerar el desarrollo de modelos de machine learning. Nuestro objetivo principal es proporcionar a los científicos de datos y a los ingenieros de machine learning una herramienta flexible y potente que les permita crear, experimentar y desplegar modelos de manera eficiente y estructurada.

## Filosofía

Nuestra filosofía se basa en tres principios fundamentales:

1. **Flexibilidad**: Creemos que cada proyecto de machine learning es único. Por eso, Framework3 está diseñado para ser altamente adaptable, permitiendo a los usuarios personalizar cada aspecto del proceso de desarrollo de modelos.

2. **Modularidad**: Hemos adoptado un enfoque modular que permite a los usuarios combinar diferentes componentes de manera sencilla. Esto facilita la experimentación y la reutilización de código.

3. **Transparencia**: Valoramos la claridad y la comprensión en cada paso del proceso de machine learning. Framework3 está diseñado para proporcionar visibilidad en todas las etapas, desde la preparación de datos hasta la evaluación del modelo.

## Arquitectura

Framework3 está construido sobre una arquitectura robusta y escalable:

### Componentes Principales

1. **Base**: Incluye las clases y tipos base que sirven como fundamento para todo el framework.

2. **Cantainer**: Proporciona estructuras para organizar y gestionar los diferentes componentes del proyecto.

3. **Plugins**: Un sistema extensible que permite añadir nuevas funcionalidades:
   - Pipelines: Para la creación de flujos de trabajo complejos.
   - Filtros: Para la transformación y procesamiento de datos.
   - Métricas: Para la evaluación de modelos.
   - Almacenamiento: Para la gestión eficiente de datos y modelos.

4. **MapReduce**: Facilita el procesamiento distribuido para manejar grandes volúmenes de datos.

5. **Utilidades**: Un conjunto de herramientas auxiliares para tareas comunes en el desarrollo de modelos.
