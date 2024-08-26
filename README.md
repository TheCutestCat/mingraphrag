

1. Place your data files in the 'data-mingraphrag/input' directory. we already have a 'Cinderella.txt' file for you.
2. Execute the following command to initialize the files: 
   ```bash
   python ./min-graphrag/main.py
   ```
3. To launch a simple RAG (Retrieval-Augmented Generation) interface based on Graphrag, run:
   ```bash
   python ./min-graphrag/UI.py
   ```

本项目是对 Graphrag 的简化实现。我们使用 `response_format` 来优化提取实体的过程，并借助 `get_stable_connected_components` 函数来计算社区（community）。由于故事相对简短，因此我们只需要一个社区，这个社区包含了所有的实体，并进行了汇总。该聚合方法与 RAPTOR 方法一致。

在 Graphrag 中，完成关系图的构建后，社区实际上代表了信息在图中的传播过程。经过多次传播，局部信息最终汇集成一个最大的社区。因此，Graphrag 更加适用于检索任务，但在生成任务方面表现相对较差。
