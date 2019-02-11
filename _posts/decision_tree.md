---
title: spark mllib 决策树算法源码学习
date: 2016-12-07
toc: true
categories: 模型与算法
tags: [决策树,spark mlilib源码]
description:  决策树算法源码学习，其中模型的训练部分以随机森林的训练过程进行说明，决策树相当于树的数量为1的随机森林
mathjax: true
---

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    extensions: ["tex2jax.js"],
    jax: ["input/TeX"],
    tex2jax: {
      inlineMath: [ ['$','$'], ['\\(','\\)'] ],
      displayMath: [ ['$$','$$']],
      processEscapes: true
    }
  });
</script>
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS_HTML,http://myserver.com/MathJax/config/local/local.js">
</script>

该文章来自于2016年后半年整理的算法源码笔记，由于之前没有写博客的习惯，都直接以笔记的形式存在电脑上，分享起来非常不便，因此抽出时间，将其整理成博客的形式，和大家一起学习交流。

# 决策树算法简要介绍

决策树算法是一种常见的分类算法，也可以用于回归问题。相对于其他分类算法，决策树的优点在于简单,可解释性强；对特征尺度不敏感，不需要做太多的特征预处理工作;能够自动挖掘特征之间的关联关系。缺点是比较容易过拟合（通过随机森林可以避免过拟合）

决策树是一个树形结构，其中叶子节点表示分类（或回归）结果，非叶子节点是属性判断判断节点，每个属性判断节点都选择样本的一个特征，并根据该特征的取值决定选择哪一个分支路径。在对样本进行预测时，从根节点开始直到叶子节点，对于路径上的每个分支节点，都根据其对应的属性取值选择下一个分支节点，直到叶子节点。整个完整的路径，表示对样本的预测过程。如图1所示，表示一个女孩在决定是否决定去相亲的一个过程，最终选择去或者不去，对应分类的结果，中间的各种条件对应相关的属性。

<center>
![“决策树样例”](decision_tree/decision_tree_example.png)
</center>
<center>图1：决策树样例：对女孩决定是否参加相亲的问题进行决策树建模</center>
 

## 决策树的训练

从根节点开始，根据信息增益或其他条件，不断选择分裂的属性，直到生成叶子节点的过程。具体过程如下所示：
* 对不同的属性，计算其信息增益，选择增益最大的特征对应根节点的最佳分裂。
* 从根节点开始，对于不同的分支节点，分别选择信息增益最大的特征作为分支节点的最佳分裂。
* 如果达到停止分裂的条件，则将该节点作为叶子节点：当前节点对应的样本都是一类样本，分类结果为对应的样本的类别；总样本数量小于一定值，或者树的高度达到最大值，或者信息增益小于一定值，或者已经用完所有的属性，选择占比最大的样本分类作为节点对应的分类结果。否则，根据步骤2进一步构造分裂节点。


## 属性度量


决策树构建的关键，在于不断地选择最佳分裂属性。属性的收益度量方法，常见的有信息增益（ID3算法）、信息增益率（C4.5算法），基尼系数(CART算法)等。

**ID3算法:**

熵：信息论中，用于描述信息的不确定性，定义如式1，其中$D$表示对样本的一个划分，$m$表示划分的类别数量，$p\_i$表示第i个类别的样本数量比例。

$info(D)=-\sum\_{i=1}^m p\_ilog\_2(p\_i)\;\;\;（式1）$

假设按照属性A对样本D进行划分，$v$为属性$A$的划分数量。则$A$对$D$划分的期望熵如式2：

$info\_A(D)=\sum\_{j=1}^v\frac{|D\_j|}{|D|}info(D\_j)\;\;\;（式2）$

信心增益为上述原始熵和属性A对D划分后期望熵的差值，可以看做是加入信息A后，不确定性的减少程度。信息增益的定义如式3所示：

$gain(A)=info(D)-info\_A(D)\;\;\;（式3）$

ID3算法即在每次选择最佳分裂的属性时，根据信息增益进行选择。

**C4.5算法:**
ID3算法容易使得选取值较多的属性。一种极端的情况是，对于ID类特征有很多的无意义的值的划分，ID3会选择该属性其作为最佳划分。C4.5算法通过采用信息增益率作为衡量特征有效性的指标，可以克服这个问题。

首先定义分裂信息：
$splitInfo\_A(D)=-\sum\_{j=1}^v\frac{|D\_j|}{|D|}log\_2(\frac{|D\_j|}{|D|})\;\;\;（式4）$

信息增益率：
$gainRatio(A)=\frac{gain(A)}{splitInfo\_A(D)}\;\;\;（式5）$

**CART算法:**

使用基尼系数作为不纯度的度量。
基尼系数:表示在样本集合中一个随机选中的样本被分错的概率，Gini指数越小表示集合中被选中的样本被分错的概率越小，也就是说集合的纯度越高，反之，集合越不纯。当所有样本属于一个类别时，基尼系数最小为0。所有类别以等概率出现时，基尼系数最大。
$GINI(P)=\sum\_{k=1}^Kp\_k(1-p\_k)=1-\sum\_{k=1}^K p\_k^2\;\;\;（式6）$

由于cart建立的树是个二叉树，所以K的取值为2。对于特征取值超过2的情况，以每个取值作为划分点，计算该划分下对应的基尼系数的期望。期望值最小的划分点，作为最佳分裂使用的特征划分。



# spark 决策树源码分析

为加深对ALS算法的理解，该部分主要分析spark mllib中决策树源码的实现。主要包括模型训练、模型预测2个部分

##  模型训练

### 决策树伴生类
    
DecisionTree伴随类，外部调用决策树模型进行训练的入口。通过外部传入数据和配置参数，调用DecisionTree中的run方法进行模型训练， 最终返回DecisionTreeModel类型对象。

```scala
object DecisionTree extends Serializable with Logging {
 def train(
      input: RDD[LabeledPoint], //训练数据，包括label和特征向量
      algo: Algo,//决策树类型，分类树or回归树
      impurity: Impurity,//衡量特征信息增益的标准，如信息增益、基尼、方差
      maxDepth: Int,//树的深度
      numClasses: Int,//待分类类别的数量
      maxBins: Int,//用于特征分裂的bin的最大数量
      quantileCalculationStrategy: QuantileStrategy,//计算分位数的算法
      //离散特征存储，如n->k表示第n个特征有k个取值（0，1，..., k-1）
      categoricalFeaturesInfo: Map[Int, Int]): DecisionTreeModel = { 
    //根据参数信息，生成决策树配置
    val strategy = new Strategy(algo, impurity, maxDepth, numClasses, maxBins,
      quantileCalculationStrategy, categoricalFeaturesInfo)
    //调用DecisionTree对象的run方法，训练决策树模型
    new DecisionTree(strategy).run(input)
  }
   //训练分类决策树
   def trainClassifier(
      input: RDD[LabeledPoint],
      numClasses: Int,
      categoricalFeaturesInfo: Map[Int, Int],
      impurity: String,
      maxDepth: Int,
      maxBins: Int): DecisionTreeModel = {
    val impurityType = Impurities.fromString(impurity)
    train(input, Classification, impurityType, maxDepth, numClasses, maxBins, Sort,categoricalFeaturesInfo)
  }
    //训练回归决策树
    def trainRegressor(
      input: RDD[LabeledPoint],
      categoricalFeaturesInfo: Map[Int, Int],
      impurity: String,
      maxDepth: Int,
      maxBins: Int): DecisionTreeModel = {
    val impurityType = Impurities.fromString(impurity) //基尼、熵、方差三种衡量标准
    train(input, Regression, impurityType, maxDepth, 0, maxBins, Sort, categoricalFeaturesInfo)
  }
}
```

### 决策树类

接受strategy参数初始化，并通过对run方法调用随机森林的run方法，通过设置特征集合为全集、树的个数为1，将随机森林训练后结果集中的第一棵树作为结果返回。

```
class DecisionTree private[spark] (private val strategy: Strategy, private val seed: Int)
  extends Serializable with Logging {
  def run(input: RDD[LabeledPoint]): DecisionTreeModel = {
    val rf = new RandomForest(strategy, numTrees = 1, featureSubsetStrategy = "all", seed = seed)
    val rfModel = rf.run(input)
    rfModel.trees(0)
  }
}
```

### RandomForest私有类run方法,通过run方法完成模型的训练

**分布式训练思想：**

*	分布式存储样本
*	对于每次迭代，算法都会对一个node集合进行分裂。对于每个node，相关worker计算的的所有相关统计特征全部传递到某个worker进行汇总，并选择最好的特征分裂
*	findSplitsBins方法可用于将连续特征离散化，在初始化阶段完成
*	迭代算法
   每次都作用于树的边缘节点，如果是随机森林，则选择所有的树的边缘节点。具体迭代步骤如下：
   1. Master 节点: 从node queue中选取节点，如果训练的是随机森林,且featureSubsetStrategy取值不是all，则对于每个节点选择随机特征子集。selectNodesToSplit用于选择待分裂的节点。
   2. Worer节点: findBestSplits函数，对每个(tree, node, feature, split)，遍历所有本地所有样本计算相关特征，计算结果通过reduceByKey传递给某个节点，由该节点汇总数据，得到(feature, split)或者判断是否停止分裂
   3. Master节点: 收集所有节点分裂信息，更新model, 并将新的model传递给各个worker节点 

#### 
```
def run(
      input: RDD[LabeledPoint],
      strategy: OldStrategy,
      numTrees: Int,
      featureSubsetStrategy: String,
      seed: Long,
      instr: Option[Instrumentation[_]],
      parentUID: Option[String] = None): Array[DecisionTreeModel] = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")
    
    val retaggedInput = input.retag(classOf[LabeledPoint])
    //构建元数据
    val metadata =
      DecisionTreeMetadata.buildMetadata(retaggedInput, strategy, numTrees, featureSubsetStrategy)
    instr match {
      case Some(instrumentation) =>
        instrumentation.logNumFeatures(metadata.numFeatures)
        instrumentation.logNumClasses(metadata.numClasses)
      case None =>
        logInfo("numFeatures: " + metadata.numFeatures)
        logInfo("numClasses: " + metadata.numClasses)
    }


    //每个特征对应的splits和bins
    timer.start("findSplits")
    val splits = findSplits(retaggedInput, metadata, seed)
    timer.stop("findSplits")
    logDebug("numBins: feature: number of bins")
    logDebug(Range(0, metadata.numFeatures).map { featureIndex =>
      s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
    }.mkString("\n"))

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    //输入
    val treeInput = TreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30,
      s"DecisionTree currently only supports maxDepth <= 30, but was given maxDepth = $maxDepth.")

    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    /*
     * The main idea here is to perform group-wise training of the decision tree nodes thus
     * reducing the passes over the data from (# nodes) to (# nodes / maxNumberOfNodesPerGroup).
     * Each data sample is handled by a particular node (or it reaches a leaf and is not used
     * in lower levels).
     */

    // Create an RDD of node Id cache.
    // At first, all the rows belong to the root nodes (node Id == 1).
    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(NodeIdCache.init(
        data = baggedInput,
        numTrees = numTrees,
        checkpointInterval = strategy.checkpointInterval,
        initVal = 1))
    } else {
      None
    }

    /*
      Stack of nodes to train: (treeIndex, node)
      The reason this is a stack is that we train many trees at once, but we want to focus on
      completing trees, rather than training all simultaneously.  If we are splitting nodes from
      1 tree, then the new nodes to split will be put at the top of this stack, so we will continue
      training the same tree in the next iteration.  This focus allows us to send fewer trees to
      workers on each iteration; see topNodesForGroup below.
     */
    val nodeStack = new mutable.Stack[(Int, LearningNode)]

    val rng = new Random()
    rng.setSeed(seed)

    // Allocate and queue root nodes.
    val topNodes = Array.fill[LearningNode](numTrees)(LearningNode.emptyNode(nodeIndex = 1))
    Range(0, numTrees).foreach(treeIndex => nodeStack.push((treeIndex, topNodes(treeIndex))))

    timer.stop("init")

    while (nodeStack.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      val (nodesForGroup, treeToNodeToIndexInfo) =
        RandomForest.selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)
      // Sanity check (should never occur):
      assert(nodesForGroup.nonEmpty,
        s"RandomForest selected empty nodesForGroup.  Error for unknown reason.")

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, LearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      RandomForest.findBestSplits(baggedInput, metadata, topNodesForGroup, nodesForGroup,
        treeToNodeToIndexInfo, splits, nodeStack, timer, nodeIdCache)
      timer.stop("findBestSplits")
    }

    baggedInput.unpersist()

    timer.stop("total")

    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    // Delete any remaining checkpoints used for node Id cache.
    if (nodeIdCache.nonEmpty) {
      try {
        nodeIdCache.get.deleteAllCheckpoints()
      } catch {
        case e: IOException =>
          logWarning(s"delete all checkpoints failed. Error reason: ${e.getMessage}")
      }
    }

    val numFeatures = metadata.numFeatures

    parentUID match {
      case Some(uid) =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.map { rootNode =>
            new DecisionTreeClassificationModel(uid, rootNode.toNode, numFeatures,
              strategy.getNumClasses)
          }
        } else {
          topNodes.map { rootNode =>
            new DecisionTreeRegressionModel(uid, rootNode.toNode, numFeatures)
          }
        }
      case None =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.map { rootNode =>
            new DecisionTreeClassificationModel(rootNode.toNode, numFeatures,
              strategy.getNumClasses)
          }
        } else {
          topNodes.map(rootNode => new DecisionTreeRegressionModel(rootNode.toNode, numFeatures))
        }
    }
  }
```




#### buildMetadata
决策树训练的元数据构造。主要用于计算每个特征的bin数量，以及无序类特征集合, 每个节点使用的特征数量等。其中决策树一般使用所有特征、随机森林分类采用$sqrt(n)$个特征，随机森林回归采用$\frac{n}{3}$个特征


```
def buildMetadata(
      input: RDD[LabeledPoint],
      strategy: Strategy,
      numTrees: Int,
      featureSubsetStrategy: String): DecisionTreeMetadata = {
    //特征数量
    val numFeatures = input.map(_.features.size).take(1).headOption.getOrElse {
      throw new IllegalArgumentException(s"DecisionTree requires size of input RDD > 0, " +
        s"but was given by empty one.")
    }
    val numExamples = input.count() //样本数量
    val numClasses = strategy.algo match {
      case Classification => strategy.numClasses
      case Regression => 0
    }
    //最大划分数量 
    val maxPossibleBins = math.min(strategy.maxBins, numExamples).toInt
    if (maxPossibleBins < strategy.maxBins) {
      logWarning(s"DecisionTree reducing maxBins from ${strategy.maxBins} to $maxPossibleBins" +
        s" (= number of training instances)")
    }
    //maxPossibleBins可能被numExamples修改过，导致小于刚开始设置的strategy.maxBins。
    //需要进一步确保离散值的特征取值数量小于maxPossibleBins，
    if (strategy.categoricalFeaturesInfo.nonEmpty) {
      val maxCategoriesPerFeature = strategy.categoricalFeaturesInfo.values.max
      val maxCategory =
        strategy.categoricalFeaturesInfo.find(_._2 == maxCategoriesPerFeature).get._1
      require(maxCategoriesPerFeature <= maxPossibleBins,
        s"DecisionTree requires maxBins (= $maxPossibleBins) to be at least as large as the " +
        s"number of values in each categorical feature, but categorical feature $maxCategory " +
        s"has $maxCategoriesPerFeature values. Considering remove this and other categorical " +
        "features with a large number of values, or add more training examples.")
    }
    //存储每个无序特征的索引
    val unorderedFeatures = new mutable.HashSet[Int]()
    //存储每个无序特征的bin数量
    val numBins = Array.fill[Int](numFeatures)(maxPossibleBins)
    if (numClasses > 2) { //多分类问题
      //根据maxPossibleBins，计算每个无序特征对应的最大类别数量
      val maxCategoriesForUnorderedFeature =
        ((math.log(maxPossibleBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
      strategy.categoricalFeaturesInfo.foreach { case (featureIndex, numCategories) =>
        //如果特征只有1个取值，则当做连续特征看待，此处对其进行过滤
          if (numCategories > 1) {
          //判断离散特征是否可当做无序特征，需要保证
          //bins的数量需要小于2 * ((1 << numCategories - 1) - 1)）
          if (numCategories <= maxCategoriesForUnorderedFeature) {
            unorderedFeatures.add(featureIndex)
            //有numCategories个取值的的特征，对应bins数量为(1 << numCategories - 1) - 1
            //此处刚开始有点疑惑，感觉应该是2 *（(1 << numCategories - 1) - 1）
            //通过DecisionTreeMetadata中numSplits函数发现，此处的bin数量和split数量有一定对应关系，(featureIndex)
           //判断划分的数量，对于无序特征, 划分数量为bin的数量；对于有序特征，为bin数量-1
            numBins(featureIndex) = numUnorderedBins(numCategories)
          } else {
            //对于其他离散特征，numBins数量为特征可能的取值数量
            numBins(featureIndex) = numCategories
          }
        }
      }
    } else { //对于二值分类或回归问题
      strategy.categoricalFeaturesInfo.foreach { case (featureIndex, numCategories) =>
        //如果特征只有1个取值，则当做连续特征看待，此处对其进行过滤
        if (numCategories > 1) {
          //numBins数量为特征可能的取值数量
          numBins(featureIndex) = numCategories 
        }
      }
    }

    //设置每个分支节点对应的特征数量
    val _featureSubsetStrategy = featureSubsetStrategy match {
      case "auto" =>
        if (numTrees == 1) { //如果是树，使用所有特征n
          "all"
        } else {
          if (strategy.algo == Classification) { //如果是用于分类的随机森林，使用sqrt(n)个特征
            "sqrt"
          } else {
            "onethird"  //如果是用于回归的随机森林，使用n/3个特征
          }
        }
      case _ => featureSubsetStrategy
    }

    val numFeaturesPerNode: Int = _featureSubsetStrategy match {
      case "all" => numFeatures
      case "sqrt" => math.sqrt(numFeatures).ceil.toInt
      case "log2" => math.max(1, (math.log(numFeatures) / math.log(2)).ceil.toInt)
      case "onethird" => (numFeatures / 3.0).ceil.toInt
      case _ =>
        Try(_featureSubsetStrategy.toInt).filter(_ > 0).toOption match {
          case Some(value) => math.min(value, numFeatures)
          case None =>
            Try(_featureSubsetStrategy.toDouble).filter(_ > 0).filter(_ <= 1.0).toOption match {
              case Some(value) => math.ceil(value * numFeatures).toInt
              case _ => throw new IllegalArgumentException(s"Supported values:" +
                s" ${RandomForestParams.supportedFeatureSubsetStrategies.mkString(", ")}," +
                s" (0.0-1.0], [1-n].")
            }
        }
    }

    new DecisionTreeMetadata(numFeatures, numExamples, numClasses, numBins.max,
      strategy.categoricalFeaturesInfo, unorderedFeatures.toSet, numBins,
      strategy.impurity, strategy.quantileCalculationStrategy, strategy.maxDepth,
      strategy.minInstancesPerNode, strategy.minInfoGain, numTrees, numFeaturesPerNode)
  }
```

#### DecisionTreeMetadata类
```  
private[spark] class DecisionTreeMetadata(
    val numFeatures: Int,
    val numExamples: Long,
    val numClasses: Int,
    val maxBins: Int,
    val featureArity: Map[Int, Int],
    val unorderedFeatures: Set[Int],
    val numBins: Array[Int],
    val impurity: Impurity,
    val quantileStrategy: QuantileStrategy,
    val maxDepth: Int,
    val minInstancesPerNode: Int,
    val minInfoGain: Double,
    val numTrees: Int,
    val numFeaturesPerNode: Int) extends Serializable {
  //判断是否为无序特征
  def isUnordered(featureIndex: Int): Boolean = unorderedFeatures.contains(featureIndex)
  //判断是否用于分类的决策树（随机森林）
  def isClassification: Boolean = numClasses >= 2
  //判断是否用于多分类的决策树（随机森林）
  def isMulticlass: Boolean = numClasses > 2
  //判断是否拥有离散特征的多分类决策树（随机森林）
  def isMulticlassWithCategoricalFeatures: Boolean = isMulticlass && (featureArity.size > 0)
  //判断是否离散特征
  def isCategorical(featureIndex: Int): Boolean = featureArity.contains(featureIndex)
 //判断是否连续特征
  def isContinuous(featureIndex: Int): Boolean = !featureArity.contains(featureIndex)
  //判断划分的数量，对于无序特征, 划分数量为bin的数量；对于有序特征，为bin数量-1
  def numSplits(featureIndex: Int): Int = if (isUnordered(featureIndex)) {
    numBins(featureIndex)
  } else {
    numBins(featureIndex) - 1
  }
  //对于连续特征，根据划分数量设置bin数量为划分数量加1
  def setNumSplits(featureIndex: Int, numSplits: Int) {
    require(isContinuous(featureIndex),
      s"Only number of bin for a continuous feature can be set.")
    numBins(featureIndex) = numSplits + 1
  }
  //判断是否需要对特征进行采样
  def subsamplingFeatures: Boolean = numFeatures != numFeaturesPerNode
}    
```

#### findSplits
通过使用采样的样本，寻找样本的划分splits和划分后的bins。

**划分的思想：**对连续特征和离散特征，分别采用不同处理方式。对于每个连续特征，numBins - 1个splits, 代表每个树的节点的所有可能的二值化分；对于每个离散特征，无序离散特征（用于多分类的维度较大的feature）基于特征的子集进行划分。有序类特征（用于回归、二分类、多分类的维度较小的feature)的每个取值对应一个bin.

```
protected[tree] def findSplits(
      input: RDD[LabeledPoint],
      metadata: DecisionTreeMetadata,
      seed: Long): Array[Array[Split]] = {
    logDebug("isMulticlass = " + metadata.isMulticlass)
    val numFeatures = metadata.numFeatures //特征的数量
    // 得到所有连续特征索引
    val continuousFeatures = Range(0, numFeatures).filter(metadata.isContinuous)
    //当有连续特征的时候需要采样样本   
    val sampledInput = if (continuousFeatures.nonEmpty) {
      // 计算近似分位数计算需要的样本数
      val requiredSamples = math.max(metadata.maxBins * metadata.maxBins, 10000)
      // 计算需要的样本占总样本比例
      val fraction = if (requiredSamples < metadata.numExamples) {
        requiredSamples.toDouble / metadata.numExamples
      } else {
        1.0
      }
      logDebug("fraction of data used for calculating quantiles = " + fraction)
      input.sample(withReplacement = false, fraction, new XORShiftRandom(seed).nextInt())
    } else {
      input.sparkContext.emptyRDD[LabeledPoint]
    }
    //对每个连续特征和非有序类离散特征，通过排序的方式，寻找最佳的splits点
    findSplitsBySorting(sampledInput, metadata, continuousFeatures)
  }
```

```
 //对每个特征，通过排序的方式，寻找最佳的splits点
 private def findSplitsBySorting(
      input: RDD[LabeledPoint],
      metadata: DecisionTreeMetadata,
      continuousFeatures: IndexedSeq[Int]): Array[Array[Split]] = {
   
    //寻找连续特征的划分阈值
    val continuousSplits: scala.collection.Map[Int, Array[Split]] = {
      //设置分区数量，如果连续特征的数量小于原始分区数，则进一步减少分区，防止无效的启动的task任务。
      val numPartitions = math.min(continuousFeatures.length, input.partitions.length)

      input
        .flatMap(point => continuousFeatures.map(idx => (idx, point.features(idx))))
        .groupByKey(numPartitions)
        .map { case (idx, samples) =>
          val thresholds = findSplitsForContinuousFeature(samples, metadata, idx)
          val splits: Array[Split] = thresholds.map(thresh => new ContinuousSplit(idx, thresh))
          logDebug(s"featureIndex = $idx, numSplits = ${splits.length}")
          (idx, splits)
        }.collectAsMap()
    }
    //特征数量
    val numFeatures = metadata.numFeatures
    //汇总所有特征的split(不包括无序离散特征)
    val splits: Array[Array[Split]] = Array.tabulate(numFeatures) {
      //如果是连续特征，返回该连续特征的split
      case i if metadata.isContinuous(i) =>
        val split = continuousSplits(i)
        metadata.setNumSplits(i, split.length)
        split
      //如果是无序离散特征，则提取该特征的split， 具体是对于每个离散特征，其第k个split为其k对应二进制的所有位置为1的数值。
      case i if metadata.isCategorical(i) && metadata.isUnordered(i) =>
        // Unordered features
        // 2^(maxFeatureValue - 1) - 1 combinations
        //特征的取值数量
        val featureArity = metadata.featureArity(i)
        Array.tabulate[Split](metadata.numSplits(i)) { splitIndex =>
          val categories = extractMultiClassCategories(splitIndex + 1, featureArity)
          new CategoricalSplit(i, categories.toArray, featureArity)
        }
      //对于有序离散特征，暂时不求解split, 在训练阶段求解
      case i if metadata.isCategorical(i) =>
        // Ordered features
        //   Splits are constructed as needed during training.
        Array.empty[Split]
    }
    splits
  }

```

```
//将input这个数对应的二进制位置为1的位置加入到当前划分
private[tree] def extractMultiClassCategories(
      input: Int,
      maxFeatureValue: Int): List[Double] = {
    var categories = List[Double]()
    var j = 0
    var bitShiftedInput = input
    while (j < maxFeatureValue) {
      if (bitShiftedInput % 2 != 0) {
        // updating the list of categories.
        categories = j.toDouble :: categories
      }
      // Right shift by one
      bitShiftedInput = bitShiftedInput >> 1
      j += 1
    }
    categories
  }
```

```
//对于连续特征，找到其对应的splits分割点
private[tree] def findSplitsForContinuousFeature(
      featureSamples: Iterable[Double], 
      metadata: DecisionTreeMetadata, 
      featureIndex: Int): Array[Double] = {
    //确保有连续特征
    require(metadata.isContinuous(featureIndex),
      "findSplitsForContinuousFeature can only be used to find splits for a continuous feature.")
    //寻找splits分割点
    val splits = if (featureSamples.isEmpty) {
      Array.empty[Double]  //如果样本数为0， 返回空数组
    } else {
      //得到metadata里的split数量
      val numSplits = metadata.numSplits(featureIndex) 

      //在采样得到的样本中，计算每个特征取值的计数、以及总样本数量
      val (valueCountMap, numSamples) = featureSamples.foldLeft((Map.empty[Double, Int], 0)) {
        case ((m, cnt), x) =>
          (m + ((x, m.getOrElse(x, 0) + 1)), cnt + 1)
      }
      // 对于每个特征取值进行排序
      val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray
      //如果得到的possible splits数量小于metadata中该特征的的split数量，则直接以当前每个特征取值作为分割的阈值
      val possibleSplits = valueCounts.length - 1
      if (possibleSplits <= numSplits) { 
        valueCounts.map(_._1).init
      } else {
        //否则，根据总样本数量，计算平均每个区间对应的特征取值数量，假设为n。然后，对于n, 2*n, 3*n ...的位置分别设置标记。设置2个游标分别指向valueCounts内部连续的两个特征取值，从前向后遍历，当后面游标到标记的距离大于前面的游标时，将前面游标的位置对应的特征取值设置为一个split点。
        //计算平均每个区间对应的特征取值数量
        val stride: Double = numSamples.toDouble / (numSplits + 1)
        logDebug("stride = " + stride)
        //splitsBuilder用于存储每个分割阈值
        val splitsBuilder = mutable.ArrayBuilder.make[Double]
        //特征取值从小到大的位置索引
        var index = 1
        //当前访问的所有特征取值数量之和
        var currentCount = valueCounts(0)._2
        //下一次的标记位置      
        var targetCount = stride
        while (index < valueCounts.length) {
          val previousCount = currentCount
          currentCount += valueCounts(index)._2
          val previousGap = math.abs(previousCount - targetCount)
          val currentGap = math.abs(currentCount - targetCount)
          //使前面游标和后面游标的距离更小，且较小游标距离标记位置的距离最近
          if (previousGap < currentGap) {
            splitsBuilder += valueCounts(index - 1)._1
            targetCount += stride
          }
          index += 1
        }
        splitsBuilder.result()
      }
    }
    splits
  }

```

#### TreePoint.convertToTreeRDD
调用TreePoint类的convertToTreeRDD方法，RDD[LabeledPoint]转化为RDD[TreePoint]。

```
 def convertToTreeRDD(
      input: RDD[LabeledPoint],
      splits: Array[Array[Split]],
      metadata: DecisionTreeMetadata): RDD[TreePoint] = {
    // 构建数组featureArity，存储每个特征对应的离散值个数，连续值对应的value为0
    val featureArity: Array[Int] = new Array[Int](metadata.numFeatures)
    var featureIndex = 0
    while (featureIndex < metadata.numFeatures) {
      featureArity(featureIndex) = metadata.featureArity.getOrElse(featureIndex, 0)
      featureIndex += 1
    }
    //获得所有连续特征的分裂阈值，如果是离散特征，则数组对应空
    val thresholds: Array[Array[Double]] = featureArity.zipWithIndex.map { case (arity, idx) =>
      if (arity == 0) {
        splits(idx).map(_.asInstanceOf[ContinuousSplit].threshold)
      } else {
        Array.empty[Double]
      }
    }
    //将样本的每个原始特征，转化为对应的bin特征值，用于训练
    input.map { x =>
      TreePoint.labeledPointToTreePoint(x, thresholds, featureArity)
    }
  }
```

```
  //将单个样本的原始特征，转化为对应的bin特征值，用于训练
  private def labeledPointToTreePoint(
      labeledPoint: LabeledPoint,
      thresholds: Array[Array[Double]],
      featureArity: Array[Int]): TreePoint = {
    //特征数量
    val numFeatures = labeledPoint.features.size
    //为每个特征找到对应的bin特征值，存储在arr数组
    val arr = new Array[Int](numFeatures)
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      //寻找数据点labeledPoint、当前特征featureIndex对应的bin特征值
      arr(featureIndex) =
        findBin(featureIndex, labeledPoint, featureArity(featureIndex), thresholds(featureIndex))
      featureIndex += 1
    }
    new TreePoint(labeledPoint.label, arr)
  }
```

```
private def findBin(
      featureIndex: Int,
      labeledPoint: LabeledPoint,
      featureArity: Int,
      thresholds: Array[Double]): Int = {
    //获取当前labeledPoint的第featureIndex个原始特征值
    val featureValue = labeledPoint.features(featureIndex)
    
    if (featureArity == 0) { 
      //如果是连续特征，利用二分法得到当前特征值对应的离散区间下标
      val idx = java.util.Arrays.binarySearch(thresholds, featureValue)
      if (idx >= 0) {
        idx
      } else {
        -idx - 1
      }
    } else {
      //如果是离散值，则直接返回当前的特征值
      if (featureValue < 0 || featureValue >= featureArity) {
        throw new IllegalArgumentException(
          s"DecisionTree given invalid data:" +
            s" Feature $featureIndex is categorical with values in {0,...,${featureArity - 1}," +
            s" but a data point gives it value $featureValue.\n" +
            "  Bad data point: " + labeledPoint.toString)
      }
      featureValue.toInt
    }
  }
```


```
//LabeledPoint类
case class LabeledPoint(@Since("2.0.0") label: Double, @Since("2.0.0") features: Vector) {
  override def toString: String = {
    s"($label,$features)"
  }
}
```
```
//TreePoint类
private[spark] class TreePoint(val label: Double, val binnedFeatures: Array[Int])
  extends Serializable {
}
```
#### BaggedPoint.convertToBaggedRDD
RDD[Datum]数据集转换成RDD[BaggedPoint[Datum]的表示类型，

```
  def convertToBaggedRDD[Datum] (
      input: RDD[Datum], //输入数据集
      subsamplingRate: Double, //采样率
      numSubsamples: Int, //采样次数
      withReplacement: Boolean, //是否有放回
      //随机数种子
      seed: Long = Utils.random.nextLong()): RDD[BaggedPoint[Datum]] = {
    if (withReplacement) {//有放回采样，生成BaggedPoint结构表示
      convertToBaggedRDDSamplingWithReplacement(input, subsamplingRate, numSubsamples, seed)
    } else {
      //当采样比为1，并且采样次数为1时，不采样，只生成BaggedPoint结构表示
      if (numSubsamples == 1 && subsamplingRate == 1.0) {
        convertToBaggedRDDWithoutSampling(input)
      } else {
        //无放回采样，生成BaggedPoint结构表示
        convertToBaggedRDDSamplingWithoutReplacement(input, subsamplingRate, numSubsamples, seed)
      }
    }
  }
```
```
  //有放回采样，数据转换为RDD[BaggedPoint[Datum]]
  private def convertToBaggedRDDSamplingWithReplacement[Datum] (
      input: RDD[Datum],//输入数据集
      subsample: Double,//采样率
      numSubsamples: Int,//采样次数
      //随机数种子
      seed: Long): RDD[BaggedPoint[Datum]] = {
    input.mapPartitionsWithIndex { (partitionIndex, instances) =>
      //每个分区生成一个泊松采样器，通过采样率、随机种子、分区索引等初始化
      val poisson = new PoissonDistribution(subsample)
      poisson.reseedRandomGenerator(seed + partitionIndex + 1)
      //将每个实例变换成BaggedPoint结构表示
      instances.map { instance =>
        val subsampleWeights = new Array[Double](numSubsamples)
        var subsampleIndex = 0
        //依次对每次采样，生成权重（即该实例在每次无放回采样出现的次数）
        while (subsampleIndex < numSubsamples) {
          subsampleWeights(subsampleIndex) = poisson.sample()
          subsampleIndex += 1
        }
        //生成BaggedPoint结构表示
        new BaggedPoint(instance, subsampleWeights) 
      }
    }
  }
```
```
//BaggedPoint类，datum表示数据实例，subsampleWeights表示当前实例在每个采样中的权重。
如(datum, [1, 0, 4])表示有3次采样，数据实例在3次采样中出现的次数分别为1，0，4
private[spark] class BaggedPoint[Datum](val datum: Datum, val subsampleWeights: Array[Double])
  extends Serializable
```
```
  //原始数据（不采样）直接转换为BaggedPoint结构表示
  private def convertToBaggedRDDWithoutSampling[Datum] (
      input: RDD[Datum]): RDD[BaggedPoint[Datum]] = {
    input.map(datum => new BaggedPoint(datum, Array(1.0)))
  }
```

```
  //无放回采样，数据转换为RDD[BaggedPoint[Datum]]
  private def convertToBaggedRDDSamplingWithoutReplacement[Datum] (
      input: RDD[Datum],
      subsamplingRate: Double,
      numSubsamples: Int,
      seed: Long): RDD[BaggedPoint[Datum]] = {
    input.mapPartitionsWithIndex { (partitionIndex, instances) =>
      //使用随机数种子，分区索引，构建随机数生成器
      val rng = new XORShiftRandom
      rng.setSeed(seed + partitionIndex + 1)
      //将每个实例变换成BaggedPoint结构表示
      instances.map { instance =>
        val subsampleWeights = new Array[Double](numSubsamples)
        var subsampleIndex = 0
        //对于每次采样，生成0-1之间的随机数，如果小于采样比，则对应权重为1，否则为0
        while (subsampleIndex < numSubsamples) {
          val x = rng.nextDouble()
          subsampleWeights(subsampleIndex) = {
            if (x < subsamplingRate) 1.0 else 0.0
          }
          subsampleIndex += 1
        }
        //转换为BaggedPoint结构数据
        new BaggedPoint(instance, subsampleWeights)
      }
    }
  }
```
#### RandomForest.selectNodesToSplit
选择当前迭代待分裂的节点，以及确定每个节点使用的特征。每次选择都根据内存限制、每个节点占用的内存（如果每个节点使用的是采样后的特征），自适应地确定节点个数。

```
private[tree] def selectNodesToSplit(
      nodeStack: mutable.Stack[(Int, LearningNode)], //存储节点的栈结构
      maxMemoryUsage: Long, //最大占用内存限制
      metadata: DecisionTreeMetadata, //元数据
      //随机数
      rng: Random): 
      //返回值包括：（1）每个树对应的待分裂节点数组， 
      //(2)每个树对应的每个节点的详细信息（包括当前分组内节点编号、特征集合）
      (Map[Int, Array[LearningNode]], Map[Int, Map[Int, NodeIndexInfo]]) = {
      //nodesForGroup(treeIndex) 存储第treeIndex个树对应的待分裂节点数组
      val mutableNodesForGroup = new mutable.HashMap[Int, mutable.ArrayBuffer[LearningNode]]()
      //每个树对应的每个节点的详细信息（包括当前分组内节点编号、特征集合）
      val mutableTreeToNodeToIndexInfo =
      new mutable.HashMap[Int, mutable.HashMap[Int, NodeIndexInfo]]()
      var memUsage: Long = 0L  //当前使用内存
      var numNodesInGroup = 0  //当前分组的节点数量
      // If maxMemoryInMB is set very small, we want to still try to split 1 node,
      // so we allow one iteration if memUsage == 0.
      //如果栈不空，并且（1）如果内存上限设置非常小，我们要去报至少能有1个节点用于分裂
      //（2）当前使用内存小于内存上限值，则进一步选择节点用于分裂
      while (nodeStack.nonEmpty && (memUsage < maxMemoryUsage || memUsage == 0)) {
      val (treeIndex, node) = nodeStack.top //选择栈顶节点
      // Choose subset of features for node (if subsampling).
     
      val featureSubset: Option[Array[Int]] = if (metadata.subsamplingFeatures) {       //如果特征需要采样，则对所有特征进行无放回采样
        Some(SamplingUtils.reservoirSampleAndCount(Range(0,
          metadata.numFeatures).iterator, metadata.numFeaturesPerNode, rng.nextLong())._1)
      } else {//如果特征不需要采样，则返回None
        None
      }
      //通过所有特征的对应的bin数量之和，以及同模型类别（分类还是回归），lable数量之间的关系确定当前节点需要使用的内存
      val nodeMemUsage = RandomForest.aggregateSizeForNode(metadata, featureSubset) * 8L
      ////检查增加当前节点后，内存容量是是否超过限制
      if (memUsage + nodeMemUsage <= maxMemoryUsage || memUsage == 0) {
        //如果加入该节点后内存没有超过限制
        nodeStack.pop() //当前节点出栈
        //更新mutableNodesForGroup，将当前节点加入对应treeIndex的节点数组
        mutableNodesForGroup.getOrElseUpdate(treeIndex, new mutable.ArrayBuffer[LearningNode]()) +=
          node
        //更新mutableTreeToNodeToIndexInfo，将当前节点的具体信息，加入对应treeindex的节点map
        mutableTreeToNodeToIndexInfo
          .getOrElseUpdate(treeIndex, new mutable.HashMap[Int, NodeIndexInfo]())(node.id)
          = new NodeIndexInfo(numNodesInGroup, featureSubset)
      }
      numNodesInGroup += 1 //当前分组的节点数量加一
      memUsage += nodeMemUsage //当前使用内存数量加一
    }
    if (memUsage > maxMemoryUsage) {
      // If maxMemoryUsage is 0, we should still allow splitting 1 node.
      logWarning(s"Tree learning is using approximately $memUsage bytes per iteration, which" +
        s" exceeds requested limit maxMemoryUsage=$maxMemoryUsage. This allows splitting" +
        s" $numNodesInGroup nodes in this iteration.")
    }
    //转换可变map为不可变map类型
    val nodesForGroup: Map[Int, Array[LearningNode]] =
      mutableNodesForGroup.mapValues(_.toArray).toMap
    val treeToNodeToIndexInfo = mutableTreeToNodeToIndexInfo.mapValues(_.toMap).toMap
    //返回（1）每个树对应的待分裂节点数组， 
    //(2)每个树对应的每个节点的详细信息（包括当前分组内节点编号、特征集合）
    (nodesForGroup, treeToNodeToIndexInfo)
  }
```

```
//无放回采样
def reservoirSampleAndCount[T: ClassTag](
      input: Iterator[T], //input输入的迭代器
      k: Int, //采样的样本数
      seed: Long = Random.nextLong()) //随机数种子
    : (Array[T], Long) = {
    val reservoir = new Array[T](k) //存储采样结果的数组
    // 放置迭代器的前k个元素到结果数组
    var i = 0
    while (i < k && input.hasNext) {
      val item = input.next()
      reservoir(i) = item
      i += 1
    }


    //如果输入元素个数小于k, 则这k个特征作为返回的结果
    if (i < k) {
      // If input size < k, trim the array to return only an array of input size.
      val trimReservoir = new Array[T](i)
      System.arraycopy(reservoir, 0, trimReservoir, 0, i)
      (trimReservoir, i) //返回结果数组，以及原始数组的元素个数
    } else { 
      //如果输入元素个数大于k, 继续采样过程，将后面元素以一定概率随机替换前面的某个元素
      var l = i.toLong
      val rand = new XORShiftRandom(seed)
      while (input.hasNext) {
        val item = input.next()
        l += 1
        //当前结果数组有k个元素，l为当前元素的序号。k/l为当前元素替换结果数组中某个元素的概率。
        //在进行替换时，对结果数组的每个元素以相等概率发生替换
        //具体方式是产生一个0到l-1之间的随机整数replacementIndex，
        //如果小于k则对第replacementIndex这个元素进行替换
        val replacementIndex = (rand.nextDouble() * l).toLong
        if (replacementIndex < k) {
          reservoir(replacementIndex.toInt) = item
        }
      }
      (reservoir, l) //返回结果数组，以及原始数组的元素个数
    }
  }
```

```
  //通过所有特征的对应的bin数量之和，以及同模型类别（分类还是回归），lable数量之间的关系确定当前节点需要使用的字节数
  private def aggregateSizeForNode(
      metadata: DecisionTreeMetadata,
      featureSubset: Option[Array[Int]]): Long = {
    //得到所有使用的特征的bin的数量之后
    val totalBins = if (featureSubset.nonEmpty) {
      //如果使用采样特征，得到采样后的所有特征bin数量之和
      featureSubset.get.map(featureIndex => metadata.numBins(featureIndex).toLong).sum
    } else {//否则使用所有的特征的bin数量之和
      metadata.numBins.map(_.toLong).sum
    }
    if (metadata.isClassification) {
      //如果是分类问题，则返回bin数量之和*类别个数
      metadata.numClasses * totalBins 
    } else {
      //否则返回bin数量之和*3
      3 * totalBins
    }
  }

```
####  RandomForest.findBestSplits

给定selectNodesToSplit方法选择的一组节点，找到每个节点对应的最佳分类特征的分裂位置。**求解的主要思想如下：**

**基于节点的分组进行并行训练：**对一组的节点同时进行每个bin的统计和计算，减少不必要的数据传输成本。这样每次迭代需要更多的计算和存储成本，但是可以大大减少迭代的次数

**基于bin的最佳分割点计算：**基于bin的计算来寻找最佳分割点，计算的思想不是依次对每个样本计算其对每个孩子节点的增益贡献，而是先将所有样本的每个特征映射到对应的bin，通过聚合每个bin的数据，进一步计算对应每个特征每个分割的增益。

**对每个partition进行聚合：**由于提取知道了每个特征对应的split个数，因此可以用一个数组存储所有的bin的聚合信息，通过使用RDD的聚合方法，大大减少通讯开销。

```
 private[tree] def findBestSplits(
      input: RDD[BaggedPoint[TreePoint]], //训练数据
      metadata: DecisionTreeMetadata, //随机森林元数据信息
      topNodesForGroup: Map[Int, LearningNode], //存储当前节点分组对应的每个树的根节点
      nodesForGroup: Map[Int, Array[LearningNode]],//存储当前节点分组对应的每个树的节点数组
      treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]],//存储当前节点分组对应的每个树索引、节点索引、及详细信息
      splits: Array[Array[Split]], //存储每个特征的所有split信息
      //存储节点的栈结构，初始化时为各个树的根节点
      nodeStack: mutable.Stack[(Int, LearningNode)],
      timer: TimeTracker = new TimeTracker,       
      nodeIdCache: Option[NodeIdCache] = None): Unit = {

    //存储当前分组的节点数量
    val numNodes = nodesForGroup.values.map(_.length).sum
    logDebug("numNodes = " + numNodes)
    logDebug("numFeatures = " + metadata.numFeatures)
    logDebug("numClasses = " + metadata.numClasses)
    logDebug("isMulticlass = " + metadata.isMulticlass)
    logDebug("isMulticlassWithCategoricalFeatures = " +
      metadata.isMulticlassWithCategoricalFeatures)
    logDebug("using nodeIdCache = " + nodeIdCache.nonEmpty.toString)

  
    //对于一个特定的树的特定节点，通过baggedPoint数据点，更新DTStatsAggregator聚合信息（更新相关的特征及bin的聚合类信息）
    def nodeBinSeqOp(
        treeIndex: Int, //树的索引
        nodeInfo: NodeIndexInfo, //节点信息
        agg: Array[DTStatsAggregator], //聚合信息，(node, feature, bin)
        baggedPoint: BaggedPoint[TreePoint]): Unit = {//数据点
      if (nodeInfo != null) {//如果节点信息不为空，表示该节点在当前计算的节点集合中
        val aggNodeIndex = nodeInfo.nodeIndexInGroup //该节点在当前分组的编号
        val featuresForNode = nodeInfo.featureSubset //该节点对应的特征集合
        //该样本在该树上的采样次数，如果为n表示5个同样的数据点同时用于更新对应的聚合信息
        val instanceWeight = baggedPoint.subsampleWeights(treeIndex) 
        if (metadata.unorderedFeatures.isEmpty) {
          //如果不存在无序特征，根据有序特征进行更新
          orderedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, instanceWeight, featuresForNode)
        } else { //都是有序特征
          mixedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, splits,
            metadata.unorderedFeatures, instanceWeight, featuresForNode)
        }
        agg(aggNodeIndex).updateParent(baggedPoint.datum.label, instanceWeight)
      }
    }

    //计算当前数据被划分到的树的节点，并更新在对应节点的聚合信息。对于每个特征的相关bin,更新其聚合信息。
    def binSeqOp(
        agg: Array[DTStatsAggregator],//agg数组存储聚合信息，数据结构为（node, feature, bin）
        baggedPoint: BaggedPoint[TreePoint]): Array[DTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        //得到要更新的节点编号
        val nodeIndex = 
          topNodesForGroup(treeIndex).predictImpl(baggedPoint.datum.binnedFeatures, splits)
        //对上步得到的节点，根据样本点更新其对应的bin的聚合信息
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }
      agg
    }

    /**
     * Do the same thing as binSeqOp, but with nodeIdCache.
     */
    def binSeqOpWithNodeIdCache(
        agg: Array[DTStatsAggregator],
        dataPoint: (BaggedPoint[TreePoint], Array[Int])): Array[DTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val baggedPoint = dataPoint._1
        val nodeIdCache = dataPoint._2
        val nodeIndex = nodeIdCache(treeIndex)
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }

      agg
    }
    
    //从treeToNodeToIndexInfo中获取每个节点对应的特征集合。key为节点在本组节点的编号，value为对应特征集合
    def getNodeToFeatures(
        treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]]): Option[Map[Int, Array[Int]]] = {
      if (!metadata.subsamplingFeatures) { //如果定义为不进行特征采样
        None
      } else {
        //定义为特征采样，从treeToNodeToIndexInfo中获取对应的节点编号和特征集合。
        val mutableNodeToFeatures = new mutable.HashMap[Int, Array[Int]]()
        treeToNodeToIndexInfo.values.foreach { nodeIdToNodeInfo =>
          nodeIdToNodeInfo.values.foreach { nodeIndexInfo =>
            assert(nodeIndexInfo.featureSubset.isDefined)
            mutableNodeToFeatures(nodeIndexInfo.nodeIndexInGroup) = nodeIndexInfo.featureSubset.get
          }
        }
        Some(mutableNodeToFeatures.toMap)
      }
    }
    
    //用于训练的节点数组
    val nodes = new Array[LearningNode](numNodes)
    //根据nodesForGroup，在nodes中存储本轮迭代的节点，存储到nodes中
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        nodes(treeToNodeToIndexInfo(treeIndex)(node.id).nodeIndexInGroup) = node
      }
    }

    //对于所有的节点，计算最佳特征及分割点
    timer.start("chooseSplits")
    //对于每个分区，迭代所有的样本，计算每个节点的聚合信息，
    //产出(nodeIndex, nodeAggregateStats)数据结构，
    //通过reduceByKey操作，一个节点的所有信息会被shuffle到同一个分区，通过合并信息，
    //计算每个节点的最佳分割，最后只有最佳的分割用于进一步构建决策树。
    val nodeToFeatures = getNodeToFeatures(treeToNodeToIndexInfo)//
    val nodeToFeaturesBc = input.sparkContext.broadcast(nodeToFeatures)

    val partitionAggregates: RDD[(Int, DTStatsAggregator)] = if (nodeIdCache.nonEmpty) {
      input.zip(nodeIdCache.get.nodeIdsForInstances).mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.map { nodeToFeatures =>
            nodeToFeatures(nodeIndex)
          }
          new DTStatsAggregator(metadata, featuresForNode)
        }
        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOpWithNodeIdCache(nodeStatsAggregators, _))
        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        nodeStatsAggregators.view.zipWithIndex.map(_.swap).iterator
      }
    } else {
      input.mapPartitions { points =>
        // 在每个分区内，构建一个nodeStatsAggregators数组，其中每个元素对应一个node的DTStatsAggregator，该DTStatsAggregator包括了决策树元数据信息、以及该node对应的特征集合
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }
          new DTStatsAggregator(metadata, featuresForNode)
        }
        //对当前分区，迭代所有样本，更新nodeStatsAggregators，即每个node对应的DTStatsAggregator
        points.foreach(binSeqOp(nodeStatsAggregators, _))
        //转化成(nodeIndex, nodeAggregateStats)格式，用于后续通过reduceByKey对多个分区的结果进行聚合。
        nodeStatsAggregators.view.zipWithIndex.map(_.swap).iterator
      }
    }
    //reduceByKey聚合多个partition的统计特征
    val nodeToBestSplits = partitionAggregates.reduceByKey((a, b) => a.merge(b)).map {
      case (nodeIndex, aggStats) =>
        //得到节点对应的特征集合
        val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
          Some(nodeToFeatures(nodeIndex))
        }

        // 找到最佳分裂特征和分裂位置，并返回度量的统计特征
        val (split: Split, stats: ImpurityStats) =
          binsToBestSplit(aggStats, splits, featuresForNode, nodes(nodeIndex))
        (nodeIndex, (split, stats))
    }.collectAsMap()

    timer.stop("chooseSplits")

    val nodeIdUpdaters = if (nodeIdCache.nonEmpty) {
      Array.fill[mutable.Map[Int, NodeIndexUpdater]](
        metadata.numTrees)(mutable.Map[Int, NodeIndexUpdater]())
    } else {
      null
    }
    // Iterate over all nodes in this group.
    //对于本组所有节点，更新节点本身信息，如果孩子节点是课分裂的叶子节点，则将其加入栈中
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        val nodeIndex = node.id //节点id
        val nodeInfo = treeToNodeToIndexInfo(treeIndex)(nodeIndex) //节点信息，包括节点在当前分组编号，节点特征等
        val aggNodeIndex = nodeInfo.nodeIndexInGroup //节点在当前分组编号
        //节点对应的最佳分裂，及最佳分裂对应的不纯度度量相关统计信息
        val (split: Split, stats: ImpurityStats) =
          nodeToBestSplits(aggNodeIndex) 
        logDebug("best split = " + split)

        //如果信息增益小于0，或者层次达到上限，则将当前节点设置为叶子节点
        val isLeaf =
          (stats.gain <= 0) || (LearningNode.indexToLevel(nodeIndex) == metadata.maxDepth)
        node.isLeaf = isLeaf
        node.stats = stats
        logDebug("Node = " + node)
        
        //当前节点非叶子节点，创建子节点
        if (!isLeaf) {
          node.split = Some(split) //设置节点split参数
          //子节点层数是否达到最大值
          val childIsLeaf = (LearningNode.indexToLevel(nodeIndex) + 1) == metadata.maxDepth
          //左孩子节点层数达到最大值，或者不纯度度量等于0，则左孩子节点为叶子节点
          val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity == 0.0)
          //右孩子节点层数达到最大值，或者不纯度度量等于0，则右孩子节点为叶子节点          
          val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity == 0.0)
          //创建左孩子节点，getEmptyImpurityStats(stats.leftImpurityCalculator)为左孩子的不纯度度量，只有impurity、impurityCalculator两个属性
          node.leftChild = Some(LearningNode(LearningNode.leftChildIndex(nodeIndex),
            leftChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.leftImpurityCalculator)))
          //创建右孩子节点
          node.rightChild = Some(LearningNode(LearningNode.rightChildIndex(nodeIndex),
            rightChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.rightImpurityCalculator)))

          if (nodeIdCache.nonEmpty) {
            val nodeIndexUpdater = NodeIndexUpdater(
              split = split,
              nodeIndex = nodeIndex)
            nodeIdUpdaters(treeIndex).put(nodeIndex, nodeIndexUpdater)
          }

          // enqueue left child and right child if they are not leaves
          //如果左孩子节点不是叶子节点，则将左孩子节点入栈
          if (!leftChildIsLeaf) {
            nodeStack.push((treeIndex, node.leftChild.get))
          }
          if (!rightChildIsLeaf) {
            //如果右孩子节点不是叶子节点，则将右孩子节点入栈
            nodeStack.push((treeIndex, node.rightChild.get))
          }
          logDebug("leftChildIndex = " + node.leftChild.get.id +
            ", impurity = " + stats.leftImpurity)
          logDebug("rightChildIndex = " + node.rightChild.get.id +
            ", impurity = " + stats.rightImpurity)
        }
      }
    }
    if (nodeIdCache.nonEmpty) {
      // Update the cache if needed.
      nodeIdCache.get.updateNodeIndices(input, nodeIdUpdaters, splits)
    }
  }  
```

```
  //得到当前数据点对应的node index输出,模仿对数据的预测过程，从根节点开始向下传播，
  //直到一个叶子节点或者未进行分裂的节点终止，返回终止节点对应的索引。
  def predictImpl(binnedFeatures: Array[Int], splits: Array[Array[Split]]): Int = {
    if (this.isLeaf || this.split.isEmpty) {
      this.id //如果当前节点是叶子节点或者未分裂的节点，返回当前节点索引
    } else {
      val split = this.split.get //当前节点的split
      val featureIndex = split.featureIndex //当前节点split对应的特征索引
      //根据数据点在featureIndex特征上的取值，以及featureIndex特征对应的分裂，判断当前数据点是否应该向左传递。
      val splitLeft = split.shouldGoLeft(binnedFeatures(featureIndex), splits(featureIndex)) 
      if (this.leftChild.isEmpty) { //如果左孩子为空
        // Not yet split. Return next layer of nodes to train
        if (splitLeft) { //当前节点应该向左传递，得到左孩子节点索引值
          LearningNode.leftChildIndex(this.id)
        } else { //当前节点应该向右传递，得到右孩子节点索引值
          LearningNode.rightChildIndex(this.id)
        }
      } else { //如果左孩子不为空，
        if (splitLeft) { //当前节点应该向左传递，从左节点开始，递归计算最终节点的索引
          this.leftChild.get.predictImpl(binnedFeatures, splits)
        } else { //当前节点应该向右传递，从右节点开始，递归计算最终节点的索引
          this.rightChild.get.predictImpl(binnedFeatures, splits)
        }
      }
    }
  }
```

```
//对于排序类特征，根据数据点、权重，更新每个特征的每个bin信息        
private def orderedBinSeqOp(
      agg: DTStatsAggregator, //聚合信息，(feature, bin)
      treePoint: TreePoint,
      instanceWeight: Double,
      featuresForNode: Option[Array[Int]]): Unit = {
    val label = treePoint.label

    // 如果是采样特征
    if (featuresForNode.nonEmpty) {
      // 使用采样的特征，对于每个特征的每个bin，进行更新
      var featureIndexIdx = 0
      while (featureIndexIdx < featuresForNode.get.length) {
        val binIndex = treePoint.binnedFeatures(featuresForNode.get.apply(featureIndexIdx))
        agg.update(featureIndexIdx, binIndex, label, instanceWeight)
        featureIndexIdx += 1
      }
    } else {
      // 如果是非采样特征，使用所有特征，对每个特征的每个bin，进行更新
      val numFeatures = agg.metadata.numFeatures
      var featureIndex = 0
      while (featureIndex < numFeatures) {
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndex, binIndex, label, instanceWeight)
        featureIndex += 1
      }
    }
  }
```

```
//相对于orderedBinSeqOp函数，mixedBinSeqOp函数在同时包括排序和非排序特征情况下，更新聚合信息.
//对于有序特征，对每个特征更新一个bin
//对于无序特征，类别的子集对应的bin需要消息，每个子集的靠左bin或者靠右bin需要更新
private def mixedBinSeqOp(
      agg: DTStatsAggregator, //聚合信息，(feature, bin)
      treePoint: TreePoint,
      splits: Array[Array[Split]],
      unorderedFeatures: Set[Int],
      instanceWeight: Double,
      featuresForNode: Option[Array[Int]]): Unit = {
    val numFeaturesPerNode = if (featuresForNode.nonEmpty) {
      // 如果特征需要采样，使用采样特征
      featuresForNode.get.length
    } else {
      // 否则使用所有特征
      agg.metadata.numFeatures
    }
    // 迭代每个特征，更新该节点对应的bin聚合信息.
    var featureIndexIdx = 0
    while (featureIndexIdx < numFeaturesPerNode) {
      //得到特征对应的原始索引值
      val featureIndex = if (featuresForNode.nonEmpty) {
        featuresForNode.get.apply(featureIndexIdx)
      } else {
        featureIndexIdx
      }
      if (unorderedFeatures.contains(featureIndex)) {
        //如果当前特征是无序特征
        val featureValue = treePoint.binnedFeatures(featureIndex) //得到bin features
        //得到当前特征偏移量
        val leftNodeFeatureOffset = agg.getFeatureOffset(featureIndexIdx)
        // Update the left or right bin for each split.
        //得到当前特征的split数量
        val numSplits = agg.metadata.numSplits(featureIndex)
        //得到当前特征分裂信息
        val featureSplits = splits(featureIndex)
        var splitIndex = 0
        while (splitIndex < numSplits) {
          //根据当前特征值，判断是否应该向左传递，如果向左传递，则将节点对当前特征的当前区间聚合信息进行更新
          if (featureSplits(splitIndex).shouldGoLeft(featureValue, featureSplits)) {
            agg.featureUpdate(leftNodeFeatureOffset, splitIndex, treePoint.label, instanceWeight)
          }
          splitIndex += 1
        }
      } else {
        // 如果是有序特征，则直接更新对应特征的对应bin信息
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndexIdx, binIndex, treePoint.label, instanceWeight)
      }
      featureIndexIdx += 1
    }
  }
```

```
//寻找最佳分裂特征和分裂位置
private[tree] def binsToBestSplit(
      binAggregates: DTStatsAggregator, //所有feature的bin的统计信息
      splits: Array[Array[Split]],//所有feature的所有split
      featuresForNode: Option[Array[Int]],//node对应的feature子集
      //当前node
      node: LearningNode): (Split, ImpurityStats) = { //返回值为最佳分裂，及对应的不纯度相关度量

    // Calculate InformationGain and ImpurityStats if current node is top node
    // 当前节点对应的树的层次
    val level = LearningNode.indexToLevel(node.id)
    // 如果是根节点，不纯度度量为0
    var gainAndImpurityStats: ImpurityStats = if (level == 0) {
      null
    } else {
      //否则为当前节点对应的相关度量stats
      node.stats
    }
    //获得合法的特征分裂
    val validFeatureSplits =
      Range(0, binAggregates.metadata.numFeaturesPerNode).view.map { 
      //得到原始特征对应的feature index
      featureIndexIdx =>
        featuresForNode.map(features => (featureIndexIdx, features(featureIndexIdx)))
          .getOrElse((featureIndexIdx, featureIndexIdx))
      }.withFilter { case (_, featureIndex) => //过滤对应split数量为0的特征
        binAggregates.metadata.numSplits(featureIndex) != 0
      }

    //对每个(feature,split), 计算增益，并选择增益最大的(feature,split)
    val (bestSplit, bestSplitStats) =
      validFeatureSplits.map { case (featureIndexIdx, featureIndex) =>
        //得到索引为featureIndex的特征对应的split数量
        val numSplits = binAggregates.metadata.numSplits(featureIndex)
        if (binAggregates.metadata.isContinuous(featureIndex)) {
          //如果是连续特征
          //计算每个bin的累积统计信息（包括第一个bin到当前bin之间的所有bin对应的统计信息）
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          var splitIndex = 0
          while (splitIndex < numSplits) {
            binAggregates.mergeForFeature(nodeFeatureOffset, splitIndex + 1, splitIndex)
            splitIndex += 1
          }
          //找到最好的split
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { case splitIdx =>
              //得到当前split左孩子对应的统计信息
              val leftChildStats = binAggregates.getImpurityCalculator(nodeFeatureOffset, splitIdx)
              //得到当前split右孩子对应的统计信息， 为得到右孩子对应的统计信息，需要所有的统计信息减去左孩子的统计信息
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, numSplits)
              //所有的统计信息减去左孩子的统计信息
              rightChildStats.subtract(leftChildStats)
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIdx, gainAndImpurityStats)//分裂索引，不纯度度量信息
            }.maxBy(_._2.gain)//取信息增益最大的分裂
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
        } else if (binAggregates.metadata.isUnordered(featureIndex)) {
          //无序离散特征
          val leftChildOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { splitIndex =>
              //得到左孩子聚合信息
              val leftChildStats = binAggregates.getImpurityCalculator(leftChildOffset, splitIndex)
              //得到右孩子聚合信息
              val rightChildStats = binAggregates.getParentImpurityCalculator()
                .subtract(leftChildStats)
              //计算不纯度度量相关统计信息
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIndex, gainAndImpurityStats) //分裂索引，不纯度度量信息
            }.maxBy(_._2.gain)//取信息增益最大的分裂
          (splits(featureIndex)(bestFeatureSplitIndex), bestFeatureGainStats)
        } else {
          // 对于排序离散特征
          //得到聚合信息的其实地址
          val nodeFeatureOffset = binAggregates.getFeatureOffset(featureIndexIdx)
          //得到类别数量
          val numCategories = binAggregates.metadata.numBins(featureIndex)

          //每个bin是一个特征值，根据质心对这些特征值排序，共K个特征值，对应生成K-1个划分
          val centroidForCategories = Range(0, numCategories).map { case featureValue =>
            //得到不纯度度量的统计信息
            val categoryStats =
              binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
            val centroid = if (categoryStats.count != 0) {//如果对应样本数量不为0，
              if (binAggregates.metadata.isMulticlass) {
                //如果是多分类决策树，则将对应多标签的不纯度度量作为质心
                categoryStats.calculate()
              } else if (binAggregates.metadata.isClassification) {
                //如果是二分类问题，则将对应的正样本数量作为质心
                categoryStats.stats(1)
              } else {
                //如果是回归问题，则将对应的预测值作为质心
                categoryStats.predict
              }
            } else {
              Double.MaxValue //如果对应样本数量为0，则质心为Double.MaxValue
            }
            (featureValue, centroid) //返回每个特征值对应的样本质心
          }

          logDebug("Centroids for categorical variable: " + centroidForCategories.mkString(","))

          // 根据质心，将特征对应的bin排序（即对应的离散特征值排序）
          val categoriesSortedByCentroid = centroidForCategories.toList.sortBy(_._2)

          logDebug("Sorted centroids for categorical variable = " +
            categoriesSortedByCentroid.mkString(","))

          // 从左到右，依次计算每个category对应的从第一个category到当前categofy的统计信息聚合结果
          var splitIndex = 0
          while (splitIndex < numSplits) {
            val currentCategory = categoriesSortedByCentroid(splitIndex)._1
            val nextCategory = categoriesSortedByCentroid(splitIndex + 1)._1
            binAggregates.mergeForFeature(nodeFeatureOffset, nextCategory, currentCategory)
            splitIndex += 1
          }
          
          //所有特征值的聚合结果对应的category索引
          val lastCategory = categoriesSortedByCentroid.last._1
          //找到最佳的分裂
          val (bestFeatureSplitIndex, bestFeatureGainStats) =
            Range(0, numSplits).map { splitIndex =>
              //得到当前索引的特征值
              val featureValue = categoriesSortedByCentroid(splitIndex)._1
              //得到左孩子对应的聚合信息
              val leftChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, featureValue)
              //得到右孩子对应的聚合信息
              val rightChildStats =
                binAggregates.getImpurityCalculator(nodeFeatureOffset, lastCategory)
              rightChildStats.subtract(leftChildStats)
              //得到不纯度度量的相关统计信息
              gainAndImpurityStats = calculateImpurityStats(gainAndImpurityStats,
                leftChildStats, rightChildStats, binAggregates.metadata)
              (splitIndex, gainAndImpurityStats)
            }.maxBy(_._2.gain)//根据信息增益进行排序，得到信息增益最大的split索引及增益
          
          //得到最佳分裂边界
          val categoriesForSplit =
            categoriesSortedByCentroid.map(_._1.toDouble).slice(0, bestFeatureSplitIndex + 1)
          //得到最佳分裂，包括特征索引、划分边界、类别数量等
          val bestFeatureSplit =
            new CategoricalSplit(featureIndex, categoriesForSplit.toArray, numCategories)
           //返回最佳分裂，及对应的增益统计信息
          (bestFeatureSplit, bestFeatureGainStats)
        }
      }.maxBy(_._2.gain)//针对所有特征，按照信息增益进行排序，取增益最大的特征

    (bestSplit, bestSplitStats)//返回最佳分裂，及对应的增益统计信息
  }
```

```
根据分裂对应的左孩子聚合信息，右孩子聚合信息，计算当前节点不纯度度量的相关统计信息
private def calculateImpurityStats(
      stats: ImpurityStats,
      leftImpurityCalculator: ImpurityCalculator,
      rightImpurityCalculator: ImpurityCalculator,
      metadata: DecisionTreeMetadata): ImpurityStats = {
    //得到父节点的聚合信息
    val parentImpurityCalculator: ImpurityCalculator = if (stats == null) {
      leftImpurityCalculator.copy.add(rightImpurityCalculator)
    } else {
      stats.impurityCalculator
    }
    //得到父节点不纯度度量
    val impurity: Double = if (stats == null) {
      parentImpurityCalculator.calculate()
    } else {
      stats.impurity
    }
   
    val leftCount = leftImpurityCalculator.count //根据当前分裂得到的左孩子对应样本数量
    val rightCount = rightImpurityCalculator.count //根据当前分裂得到的右孩子对应样本数量

    val totalCount = leftCount + rightCount  //当前分裂对应的总样本数量

    // If left child or right child doesn't satisfy minimum instances per node,
    // then this split is invalid, return invalid information gain stats.
    //如果左孩子或者右孩子样本数量小于下限值，返回不合法的不纯度度量信息
    if ((leftCount < metadata.minInstancesPerNode) ||
      (rightCount < metadata.minInstancesPerNode)) {
      return ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
    }
    //左孩子对应的不纯度度量
    val leftImpurity = leftImpurityCalculator.calculate() // Note: This equals 0 if count = 0
    //右孩子对应的不纯度度量
    val rightImpurity = rightImpurityCalculator.calculate()
    //左孩子权重
    val leftWeight = leftCount / totalCount.toDouble
    //右孩子权重
    val rightWeight = rightCount / totalCount.toDouble
    //信息增益
    val gain = impurity - leftWeight * leftImpurity - rightWeight * rightImpurity
    //信息增益小于下限值，则返回不合法的不纯度度量信息
      if (gain < metadata.minInfoGain) {
      return ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator)
    }
    //返回不纯度度量信息
    new ImpurityStats(gain, impurity, parentImpurityCalculator,
      leftImpurityCalculator, rightImpurityCalculator)
  }
```

## 模型预测

通过模型训练生成决策树（随机森林）模型RandomForestModel，随机森林模型继承了树的组合模型TreeEnsembleModel，进一步通过predictBySumming函数，对传进的样本点进行预测。


```
  //对样本点features进行预测
  private def predictBySumming(features: Vector): Double = {
    //对每棵决策树进行预测，然后自后结果为每个决策树结果的加权求和
    val treePredictions = trees.map(_.predict(features))
    blas.ddot(numTrees, treePredictions, 1, treeWeights, 1)
  }
  
```

```
  //DecisionTreeModel.predict方法
  def predict(features: Vector): Double = {
    //根据头部节点预测lable
    topNode.predict(features)
  }
```

```
  //Node. predict方法
  def predict(features: Vector): Double = {
    if (isLeaf) {
      predict.predict //如果是叶子节点，直接输出
    } else {
      if (split.get.featureType == Continuous) { 
        //如果是连续特征，根据分裂阈值，决定走左孩子节点还是右孩子节点
        if (features(split.get.feature) <= split.get.threshold) {
          leftNode.get.predict(features)
        } else {
          rightNode.get.predict(features)
        }
      } else {
        //如果是离散特征，根据特征是否被当前节点对应的特征集合包含，决定走左孩子节点还是右孩子节点
        if (split.get.categories.contains(features(split.get.feature))) {
          leftNode.get.predict(features)
        } else {
          rightNode.get.predict(features)
        }
      }
    }
  }

```


# 参考资料

【1】http://spark.apache.org/mllib/ 
【2】http://www.cnblogs.com/leoo2sk/archive/2010/09/19/decision-tree.html
