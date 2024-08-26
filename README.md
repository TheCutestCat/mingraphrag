

1. Place your data files in the 'data-mingraphrag/input' directory.
2. Execute the following command to initialize the files: 
   ```bash
   python ./mingraphrag/main.py
   ```
3. To launch a simple RAG (Retrieval-Augmented Generation) interface based on Graphrag, run:
   ```bash
   python ./mingraphrag/UI.py
   ```

这个是对graphrag的一个简单模仿，我们使用response_format来简化graphrag中提取实体的步骤，同时借用get_stable_connected_components函数来计算其中的community，因为故事比较短所以我们只有一个群组，也就是将所有的实体放进去然后总结了一遍。。 对应的聚合方法和RAPTOR方法一致。

graphrag在创建完成关系图之后，community实际上就是图在信息上的传播，经过多次传播之后局部的信息就汇总到了一个最大的community上。
因此其更加适合检索任务，对于生成任务之类的效果很差。

