---
title: ALS推荐算法学习与实践
toc: true
categories: machine-learning
tags: 推荐算法、协同过滤、矩阵分解
description: als算法原理及源码学习
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


ALS（alternating least squares）是一种基础的推荐算法，相对于普通的协同过滤等方法，它不仅能通过降维增加模型的泛化能力，也方便加入其他建模因素（如时间、隐反馈及置信度等），大大提升了模型的灵活性。正因为此，ALS算法在Netflix推荐大赛中脱颖而出，在我们具体的工程实践中，也具有非常不错的表现。接下来，从如下几个方面和大家一起学习：ALS算法模型、spark ALS源码理解， ALS推荐实践。如描述有误，欢迎大家指正。

# ALS算法模型
## 为什么要用ALS模型
 相对于其他模型，ALS模型优势如下：
 
* **相对于基于内容的推荐**，ALS属于协同过滤大家族【1】（也有人认为ALS 基于矩阵分解技术，不属于协同过滤范畴【2】），**直接跟进用户行为信息进行建模，不需要获取user和item的内容信息**（很多情况下这些内容信息并不是很好获取，但是相对基于内容的推荐，ALS存在冷启动问题）

* **相对于传统的协同过滤推荐方法（user based、item based）**， ALS算法属于factor model, 通过将数据从原始空间映射到更低维度空间，**去除噪声信息，利用更主要的语义信息对问题建模，能获得更好的推荐效果**。

* **相对于svd分解模型而言**， 两种模型都属于 factor model, 但**svd分解模型更倾向于解决矩阵元素没有缺失的情况， 而通过一定的方式去填充矩阵不仅需要额外的填充成本，填充本身可能影响了数据的真实性**。因此，直接对已知元素进行建模，是一种不错的思路。如【1，3-6】，直接对rating矩阵已知元素$r\_{ui}$进行建模:

<center>
$\sum\_{u,i\in\mathbb K} (r\_{ui} - 
p\_u^Tq\_i)^2 + \lambda(p\_u^Tp\_u+q\_i^Tq\_i)$ （1）
</center>

* 针对所建模型1可以用SGD或ALS 两种算法求解。其中**sgd方法相对比较简单，但是当我们要建模的矩阵中已知元素较多时（如隐反馈），采用sgd在每次迭代都要求解所有元素，其时间复杂度是非常大的**。ALS算法在求解某个user （或item）向量时，不依赖其他任何user（item）向量，这个性质使得**ALS算法在每次迭代过程中方便并行化求解，在解决大规模矩阵分解问题时非常具有优势**。 
  
## ALS模型是什么
### 基本概念

ALS模型属于隐语义模型，通过对用户行为矩阵R进行矩阵分解，得到user factor向量矩阵P、item factor向量矩阵Q. 

$R = P^T Q$ 。其中R、$P^T$、$Q^T$矩阵的定义如表1-表3所示。
    
潜在语义空间对应的各个factor代表不同的属性信息，user向量描述了user对各种属性的喜好程度，item向量描述了item所具备的各种属性强度，二者在潜在语义空间的相似度描述了user对item的喜好程度,在进行推荐时，根据该喜好程度计算推荐结果。
<center>表1: rating矩阵R</center>

 |item1|item2|item3|item4
-|-|-|-|-|
user1|$r\_{11}$|$r\_{12}$| $r\_{13}$| $r\_{14}$
user2|$r\_{21}$|$r\_{22}$| $r\_{23}$| $r\_{24}$
user3|$r\_{31}$|$r\_{32}$| $r\_{33}$| $r\_{34}$
user4|$r\_{41}$|$r\_{42}$| $r\_{43}$| $r\_{44}$
user5|$r\_{51}$|$r\_{52}$| $r\_{53}$| $r\_{54}$
 

<center>表2：user矩阵$P^T$</center>

 | factor1 | factor2 | factor3 
-|-|-|-|
user1|$p\_{11}$|$p\_{12}$| $p\_{13}$
user2|$p\_{21}$|$p\_{22}$| $p\_{23}$
user3|$p\_{31}$|$p\_{32}$| $p\_{33}$
user4|$p\_{41}$|$p\_{42}$| $p\_{43}$
user5|$p\_{51}$|$p\_{52}$| $p\_{53}$


<center>表3:item矩阵$Q^T$</center>

 | factor1 | factor2 | factor3 
-|-|-|-|
item1|$q\_{11}$|$q\_{12}$| $q\_{13}$
item2|$q\_{21}$|$q\_{22}$| $q\_{23}$
item3|$q\_{31}$|$q\_{32}$| $q\_{33}$
item4|$q\_{41}$|$q\_{42}$| $q\_{43}$


### 目标函数
  
$MIN\_{PQ} \sum\_{u,i\in\mathbb K} {(r\_{ui} - 
p\_u^Tq\_i）}^2 + \lambda(p\_u^Tp\_u+q\_i^Tq\_i)$  (2)

其中${(r\_{ui} - p\_u^Tq\_i）}^2$ 目的在于最小化分解误差，$\lambda(p\_u^Tp\_u+q\_i^Tq\_i)$ 为正则项。

### 目标函数求解

由于目标函数中$p\_u, q_i$都是未知变量，该问题是非凸的。当我们固定其中一个变量，解另外一个变量时，问题则变成凸问题，这是ALS求解的主要思想。在实际求解过程中分为如下几个步骤：

1. 随机初始化所有的变量$p\_u, q\_i$。
  
2. 固定所有的$q\_i$变量，求得$q\_i$变量为当前值时$p\_u$的最优值。
  
3. 固定所有的$p\_u$变量，求得$p\_u$变量为当前值时$q\_i$的最优值。
  
4. 如果满足终止条件，则终止。否则，迭代执行2，3两步。

通过不断执行步骤2和步骤3，使得误差越来越小，直到收敛或达到指定次数而终止。通过可导函数性质我们知道，当对变量求导结果等于0当时候，函数可以取得极值。具体到公式2，固定一个变量，对另一变量求导结果等于0时，可以达到极小值。
 
我们令$L = \sum\_{u,i\in\mathbb K} {(r\_{ui} - p\_u^Tq\_i）}^2 + \lambda(p\_u^Tp\_u+q\_i^Tq\_i)$

固定所有$q\_i$, 对$p\_u$求导
  
$-\frac{\alpha L}{2\alpha p\_{uk}} = \sum\_{i} {q\_{ik}(r\_{ui} - p\_u^Tq\_i）} - \lambda p\_{uk} = 0$

=> $\sum\_{i} {q\_{i}(r\_{ui} - p\_u^Tq\_i）} - \lambda p\_{u} = 0$

=> $(\sum\_{i} {q\_i q\_i^T} + \lambda E) p\_u = \sum\_{i}q\_i r\_{ui}$

=> $p\_u = (\sum\_{i} {q\_i q\_i^T} + \lambda E)^{-1}\sum\_{i}q\_i r\_{ui}$

=> $ p\_u = (Q\_{u,i\in\mathbb K} Q\_{u,i\in\mathbb K}^T + \lambda E)^{-1}Q\_{u,i\in\mathbb K}R\_{u,i\in\mathbb K}^T$

其中，$q\_{u,i\in\mathbb K}$ 表示和user $u$有行为关联的item对应的向量矩阵，$r\_{u,i\in\mathbb K}^T$表示和user $u$有行为关联的item对应rating元素构成的向量的转置。

**更加灵活的ALS建模**

相对于传统的协同协同过滤方法，ALS能更好的考虑其他因素，如数据偏差、时间等

1. 引入数据偏差
    
    user偏差：不同的用户，可能具有不同的评分标准。如用户在给item打分时，有的用户可能可能更倾向于给所有item打高分， 而有的挑剔用户会给所有item打分偏低
    
    item偏差：有的热门item可能所有用户都会倾向于打高分，而有的item可能本身大多数人会倾向于打低分
    
    考虑use和item偏差的ALS建模：
    $MIN\_{PQB} \sum\_{u,i\in\mathbb K} {(r\_{ui} - u - b\_u - b\_i-
p\_u^Tq\_i）}^2 + \lambda(p\_u^Tp\_u+q\_i^Tq\_i+b\_u^2+b\_i^2)$  (3)

2. 引入时间因素
    
    用户偏好、rating矩阵，都可能随时间变化，item对应的属性不随时间变化，因此可进行如下建模
$MIN\_{PQB} \sum\_{u,i\in\mathbb K} {(r\_{ui}（t） - u - b\_u(t) - b\_i(t)-
p\_u(t)^Tq\_i）}^2 + \lambda(p\_u(t)^Tp\_u(t)+q\_i^Tq\_i+b\_u(t)^2+b\_i(t)^2)$  (4)
    
3. 引入隐反馈数据因素
    
    很多时候，并没有用户对item明确的打分数据，此时可通过搜集用户隐反馈数据（浏览、点击、点赞等），进行隐反馈建模。有一点需要注意，此时不只是对$r\_{ui}$大于0对用户行为建模，而是所有$r\_{ui}$元素建模。模型如公式5所示：
    
    $MIN\_{PQB} \sum\_{u,i} {c\_{ui}(p\_{ui} - u - b\_u - b\_i-
p\_u^Tq\_i）}^2 + \lambda(p\_u^Tp\_u+q\_i^Tq\_i+b\_u^2+b\_i^2)$  (5)
 
    $p\_{ui}$ 表示user u是否有相关行为表示喜欢item i, $c\_{ui}$描述user u 对item i的喜欢程度，其定义如公式6和公式7所示
 
    $
p\_{ui} = 
\begin{cases} 
1,  & r\_{ui}>0\\\\
0,  & r\_{ui}=0
\end{cases}
$（6）
    
    $c\_{ui} = 1 + \alpha r\_{ui}$（7）



# spark ALS源码理解
    
为加深对ALS算法的理解，该部分主要分析spark mllib中ALS源码的实现，大体上分为2部分：ALS模型训练、ALS模型推荐

## ALS 模型训练

### ALS 伴生类
    
ALS 伴生对象提供外部调用 ALS模型训练的入口。通过传入相关参数， 返回训练好的模型对象MatrixFactorizationModel。


```scala
object ALS {
  def train(
      ratings: RDD[Rating], //rating元素 （user, item, rate）
      rank: Int, //隐语义个数
      iterations: Int, //迭代次数
      lambda: Double, //正则惩罚项
      blocks: Int, //数据block个数
      seed: Long //随机数种子
    ): MatrixFactorizationModel = {
    new ALS(blocks, blocks, rank, iterations, lambda, fALSe, 1.0, seed).run(ratings)
  }

  def trainImplicit(
      ratings: RDD[Rating], // rating元素 （user, item, rate）
      rank: Int, //隐语义个数
      iterations: Int, //迭代次数
      lambda: Double, //正则惩罚项
      blocks: Int, //数据block个数
      alpha: Double //计算$c_ui$时用的alpha参数
    ): MatrixFactorizationModel = {
    new ALS(blocks, blocks, rank, iterations, lambda, true, alpha).run(ratings)
  }
   //另外还有一些其他接口，因最终都通过调用上面2个函数，此处将其省略
}
```

### ALS 私有类
    
定义了ALS类对应的各个参数，以及各个参数的设定方法。并定义了run方法供伴随类进行调用，该方法返回训练结果MatrixFactorizationModel给ALS伴随类。

    
```scala
class ALS private (
    private var numUserBlocks: Int, //用户数据block个数
    private var numProductBlocks: Int, //item数据block个数
    private var rank: Int, //隐语义个数
    private var iterations: Int, //迭代次数
    private var lambda: Double, //正则惩罚项
    private var implicitPrefs: Boolean, //是否使用隐反馈模型
    private var alpha: Double, //计算$c_ui$时用的alpha参数
    private var seed: Long = System.nanoTime() //随机数种子,默认为当前时间戳
  ) extends Serializable with Logging {

  //设置block个数
  def setBlocks(numBlocks: Int): this.type = {
    this.numUserBlocks = numBlocks
    this.numProductBlocks = numBlocks
    this
  }
  
  // 另外对其他参数变量也有相关函数实现，因基本都是赋值操作，此处将其省略
  
  //run方法，通过输入rating数据，完成训练兵返回结果MatrixFactorizationModel
  def run(ratings: RDD[Rating]): MatrixFactorizationModel = {
    require(!ratings.isEmpty(), s"No ratings available from $ratings")

    val sc = ratings.context
    //设置user block个数
    val numUserBlocks = if (this.numUserBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.length / 2)
    } else {
      this.numUserBlocks
    }
    //设置item block个数
    val numProductBlocks = if (this.numProductBlocks == -1) {
      math.max(sc.defaultParallelism, ratings.partitions.length / 2)
    } else {
      this.numProductBlocks
    }
    //调用NewALS.train方法完成矩阵分解，生成user factor和item factor向量,该方法是整个ALS算法的核心实现
    val (floatUserFactors, floatProdFactors) = NewALS.train[Int](
      ratings = ratings.map(r => NewALS.Rating(r.user, r.product, r.rating.toFloat)),
      rank = rank,
      numUserBlocks = numUserBlocks,
      numItemBlocks = numProductBlocks,
      maxIter = iterations,
      regParam = lambda,
      implicitPrefs = implicitPrefs,
      alpha = alpha,
      nonnegative = nonnegative,
      intermediateRDDStorageLevel = intermediateRDDStorageLevel,
      finalRDDStorageLevel = StorageLevel.NONE,
      checkpointInterval = checkpointInterval,
      seed = seed)
   
    val userFactors = floatUserFactors
      .mapValues(_.map(_.toDouble))
      .setName("users")
      .persist(finalRDDStorageLevel)
    val prodFactors = floatProdFactors
      .mapValues(_.map(_.toDouble))
      .setName("products")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userFactors.count()
      prodFactors.count()
    }
    //生成和返回ALS模型 MatrixFactorizationModel
    new MatrixFactorizationModel(rank, userFactors, prodFactors)
  }
}
```


### NewALS.train方法
    
被ALS私有类的run方法调用，用于计算user factor和item factor向量。

```scala
def train[ID: ClassTag]( // scalastyle:ignore
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      regParam: Double = 1.0,
      implicitPrefs: Boolean = fALSe,
      alpha: Double = 1.0,
      nonnegative: Boolean = fALSe,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      checkpointInterval: Int = 10,
      seed: Long = 0L)(
      implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = {
    require(!ratings.isEmpty(), s"No ratings available from $ratings")
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext
    //根据block个数，构建哈稀器。
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    //构建索引编码器，根据block编号和block内索引进行编码，同时可将编码后结果快速解码为block编号和block内索引号。具体实现是通过block个数，确定block编码需要的二进制位数，以及block内索引位数，通过这些位数利用逻辑操作即可实现编码和解码

    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
    //构建求解器
    val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
    //对rating矩阵进行分块，得到((user_blockID, item_blockID),rating(user, item, rating))

    val blockRatings = partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
    //构建user inblock和outblock数据，inblock数据记录每个user对应的所有item的地址，及对应rating信息。 outblock记录当前block的哪些user数据会被哪些block用上
    val (userInBlocks, userOutBlocks) =
      makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    // materialize blockRatings and user blocks
    userOutBlocks.count()
    //交换blockrating中的user, item数据，用于构造item的inblcok和outblock信息
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks) =
      makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    // materialize item blocks
    itemOutBlocks.count()
    val seedGen = new XORShiftRandom(seed)
    //随机初始化user factor和item factor
    var userFactors = initialize(userInBlocks, rank, seedGen.nextLong())
    var itemFactors = initialize(itemInBlocks, rank, seedGen.nextLong())
    var previousCheckpointFile: Option[String] = None
    val shouldCheckpoint: Int => Boolean = (iter) =>
      sc.checkpointDir.isDefined && checkpointInterval != -1 && (iter % checkpointInterval == 0)
    val deletePreviousCheckpointFile: () => Unit = () =>
      previousCheckpointFile.foreach { file =>
        try {
          val checkpointFile = new Path(file)
          checkpointFile.getFileSystem(sc.hadoopConfiguration).delete(checkpointFile, true)
        } catch {
          case e: IOException =>
            logWarning(s"Cannot delete checkpoint file $file:", e)
        }
      }
    //针对隐反馈，迭代求解
    if (implicitPrefs) {
      for (iter <- 1 to maxIter) {  //迭代总次数maxIter
        userFactors.setName(s"userFactors-$iter").persist(intermediateRDDStorageLevel)
        val previousItemFactors = itemFactors
        //固定user factor，优化item factor
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, implicitPrefs, alpha, solver)
        previousItemFactors.unpersist()
        itemFactors.setName(s"itemFactors-$iter").persist(intermediateRDDStorageLevel)
        // TODO: Generalize PeriodicGraphCheckpointer and use it here.
        val deps = itemFactors.dependencies
        if (shouldCheckpoint(iter)) {
          itemFactors.checkpoint() // itemFactors gets materialized in computeFactors
        }
        val previousUserFactors = userFactors
        //根据item factore, 优化user factor
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, implicitPrefs, alpha, solver)
        if (shouldCheckpoint(iter)) {
          ALS.cleanShuffleDependencies(sc, deps)
          deletePreviousCheckpointFile()
          previousCheckpointFile = itemFactors.getCheckpointFile
        }
        previousUserFactors.unpersist()
      }
    } else { //针对显示反馈，迭代求解
      for (iter <- 0 until maxIter) { //迭代总次数maxIter
        //固定user factor，优化item factor
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, solver = solver)
        if (shouldCheckpoint(iter)) {
          val deps = itemFactors.dependencies
          itemFactors.checkpoint()
          itemFactors.count() // checkpoint item factors and cut lineage
          ALS.cleanShuffleDependencies(sc, deps)
          deletePreviousCheckpointFile()
          previousCheckpointFile = itemFactors.getCheckpointFile
        }
        //根据item factore, 优化user factor
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, solver = solver)
      }
    }
    //将user id 和 factor拼接在一起
    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
      // and userFactors.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    //将item id 和 factor拼接在一起
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemFactors.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    //返回user factor和item factor数据
    (userIdAndFactors, itemIdAndFactors)
  }
```

### 构建哈希器 
  
   构建哈希器，用于计算user或item id对应的block编号。
    
```scala
class HashPartitioner(partitions: Int) extends Partitioner {
  require(partitions >= 0, s"Number of partitions ($partitions) cannot be negative.")
  //block总数
  def numPartitions: Int = partitions
  //通过求余计算block 编号
  def getPartition(key: Any): Int = key match {
    case null => 0
    case _ => Utils.nonNegativeMod(key.hashCode, numPartitions)
  }
  //判断2个哈希器是否相等
  override def equALS(other: Any): Boolean = other match {
    case h: HashPartitioner =>
      h.numPartitions == numPartitions
    case _ =>
      fALSe
  }
  override def hashCode: Int = numPartitions
}
```

### 构建地址编码解码器

构建地址编码解码器，根据block编号和block内索引对地址进行编码，同时可将编码后地址解码为block编号和block内索引号。具体实现是通过block个数确定block编码需要的二进制位数，以及block内索引位数，通过这些位数利用逻辑操作即可实现地址的编码和解码。
    
```scala
private[recommendation] class LocalIndexEncoder(numBlocks: Int) extends Serializable {

    require(numBlocks > 0, s"numBlocks must be positive but found $numBlocks.")
    //block内部索引使用的二进制位数
    private[this] final val numLocalIndexBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(numBlocks - 1), 31)
    private[this] final val localIndexMask = (1 << numLocalIndexBits) - 1
    
    //根据block编号和block内索引值，对地址编码
    def encode(blockId: Int, localIndex: Int): Int = {
      require(blockId < numBlocks)
      require((localIndex & ~localIndexMask) == 0)
      (blockId << numLocalIndexBits) | localIndex
    }

    //根据编码后地址，得到block编号
    def blockId(encoded: Int): Int = {
      encoded >>> numLocalIndexBits
    }

    //根据编码地址，得到block内部索引
    def localIndex(encoded: Int): Int = {
      encoded & localIndexMask
    }
  }
```

### partition rating

格式化rating数据，将rating数据分块，根据user和product的id哈希后的结果，得到对应的块索引。最终返回（src_block_id, dst_block_id）(src_id数组，dst_id数组，rating数组)

    
```scala
private def partitionRatings[ID: ClassTag](
      ratings: RDD[Rating[ID]],
      srcPart: Partitioner,
      dstPart: Partitioner): RDD[((Int, Int), RatingBlock[ID])] = {
    //获得总block数
    val numPartitions = srcPart.numPartitions * dstPart.numPartitions
    //在rating的每个分区，计算每个rating元素对应的src_block_id和dst_block_id, 并放到对应的块索引中。然后，对所有分区的元素按照块索引进行聚合，并返回聚合结果
    ratings.mapPartitions { iter =>
      //生成numPartitions个一维数组，存储对应block的rating记录
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap { r =>
        val srcBlockId = srcPart.getPartition(r.user) //user block id
        val dstBlockId = dstPart.getPartition(r.item) //item block id
        val idx = srcBlockId + srcPart.numPartitions * dstBlockId //数组索引计算
        //将对应的rating元素放在builders对应元素中
        val builder = builders(idx) 
        builder.add(r) 
        if (builder.size >= 2048) { // 2048 * (3 * 4) = 24k
          //如果某个block内数据量较多，直接得到结果
          builders(idx) = new RatingBlockBuilder
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        //对builders数组内元素，计算对应的src_block_id和dst_block_id,并将对应rating数据放在其中
        builders.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val srcBlockId = idx % srcPart.numPartitions
          val dstBlockId = idx / srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { blocks =>
      //对不同分区计算出的的rating元素进行聚合
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build() //value为 （src_id数组，dst_id数组，对应的rating数组）
    }.setName("ratingBlocks")
  }

```

### 构造in_block, 和out_block
  
在分布式计算中，不同节点的通信是影响程序效率重要原因，通过合理的设计分区，使得不同节点交换数据尽量少，可以有效的提升运行效率。
     
由上述章节中对目标函数求解推导，可以得知，每个用户向量的计算依赖于所有和它关联的item向量。如果不做任何优化，则每次优化user向量时，所有user向量的计算，都需要从其他节点得到对应item向量。如果节点A上有多个user和节点B上的某一item关联，则节点B需要向节点A传输多次item向量数据，实际上这是不必要的。优化的思路是，通过合理的分区，提前计算好所有节点需要从其它节点获取的item向量数据，将其缓存在本地，计算每个user向量时，直接从本地读取，可以大大减少需要传输的数据量，提升程序执行的效率。
     
在源码中，通过out block缓存当前节点需要向其它节点传输的数据， in block用于缓存当前节点需要的数据索引。当其他节点信息传输到本地时，通过读取in block内索引信息，来从本地获取其它节点传过来的数据。更加详细的描述可参考【7】
    
in block 结构： （block_id, Inblock(src_id数组, src_ptr, dst_id地址数组， rating数组）)
out block结构： （block_id， array[array[int]]） （二维数组存储发往每个block的src_id索引）

    
```scala
private def makeBlocks[ID: ClassTag](
      prefix: String,
      ratingBlocks: RDD[((Int, Int), RatingBlock[ID])],
      srcPart: Partitioner,
      dstPart: Partitioner,
      storageLevel: StorageLevel)(
      implicit srcOrd: Ordering[ID]): (RDD[(Int, InBlock[ID])], RDD[(Int, OutBlock)]) = {
    //根据ratingBlocks.map计算inBlocks
    val inBlocks = ratingBlocks.map {
      case ((srcBlockId, dstBlockId), RatingBlock(srcIds, dstIds, ratings)) =>
        val start = System.nanoTime()
        //dst id去重复
        val dstIdSet = new OpenHashSet[ID](1 << 20) 
        dstIds.foreach(dstIdSet.add)  
        //dst id 去重结果进行排序
        val sortedDstIds = new Array[ID](dstIdSet.size)
        var i = 0
        var pos = dstIdSet.nextPos(0)
        while (pos != -1) {
          sortedDstIds(i) = dstIdSet.getValue(pos)
          pos = dstIdSet.nextPos(pos + 1)
          i += 1
        }
        assert(i == dstIdSet.size)
        Sorting.quickSort(sortedDstIds)
        //得到dst id 对应的去重和排序后的索引值
        val dstIdToLocalIndex = new OpenHashMap[ID, Int](sortedDstIds.length)
        i = 0
        while (i < sortedDstIds.length) {
          dstIdToLocalIndex.update(sortedDstIds(i), i)
          i += 1
        }
        logDebug(
          "Converting to local indices took " + (System.nanoTime() - start) / 1e9 + " seconds.")
        val dstLocalIndices = dstIds.map(dstIdToLocalIndex.apply)
        (srcBlockId, (dstBlockId, srcIds, dstLocalIndices, ratings))
    }.groupByKey(new ALSPartitioner(srcPart.numPartitions)) //根据src block id进行聚合
      .mapValues { iter =>
        val builder =
          new UncompressedInBlockBuilder[ID](new LocalIndexEncoder(dstPart.numPartitions))
        //将dstBlockId和dstLocalIndices编码，并汇总数据
        iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
          builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
        }
        //对结果进行压缩存储，结果格式为（uniqueSrcId数组, dstPtrs数组, dstEncodedIndices数组, ratings数组）
        builder.build().compress()
      }.setName(prefix + "InBlocks")
      .persist(storageLevel)
        
    //根据inBlocks计算outBlocks
    val outBlocks = inBlocks.mapValues { case InBlock(srcIds, dstPtrs, dstEncodedIndices, _) =>
      //构造编码器
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      //定义ArrayBuilder数组，存储发往每个out block的 src id信息
      val activeIds = Array.fill(dstPart.numPartitions)(mutable.ArrayBuilder.make[Int])
      var i = 0
      val seen = new Array[Boolean](dstPart.numPartitions)
      //依次计算当前src id是否发往每一个block id
      while (i < srcIds.length) {
        var j = dstPtrs(i)
        ju.Arrays.fill(seen, fALSe)
        while (j < dstPtrs(i + 1)) {
          val dstBlockId = encoder.blockId(dstEncodedIndices(j))
          if (!seen(dstBlockId)) {
            activeIds(dstBlockId) += i // add the local index in this out-block
            seen(dstBlockId) = true
          }
          j += 1
        }
        i += 1
      }
      activeIds.map { x =>
        x.result()
      }
    }.setName(prefix + "OutBlocks")
      .persist(storageLevel)
    (inBlocks, outBlocks)  //返回结果
  }
```

#### inblock compress

  对inblock 中间结果压缩存储，返回结果格式为（uniqueSrcId数组, dstPtrs数组, dstEncodedIndices数组, ratings数组）

    
```scala
def compress(): InBlock[ID] = {
  val sz = length
  assert(sz > 0, "Empty in-link block should not exist.")
  sort()
  val uniqueSrcIdsBuilder = mutable.ArrayBuilder.make[ID]
  val dstCountsBuilder = mutable.ArrayBuilder.make[Int]
  var preSrcId = srcIds(0)
  uniqueSrcIdsBuilder += preSrcId
  var curCount = 1
  var i = 1
  var j = 0
  //得到去重后的src id数组， 以及每个src id的数量
  while (i < sz) {
    val srcId = srcIds(i)
    if (srcId != preSrcId) {
      uniqueSrcIdsBuilder += srcId
      dstCountsBuilder += curCount
      preSrcId = srcId
      j += 1
      curCount = 0
    }
    curCount += 1
    i += 1
  }
  dstCountsBuilder += curCount
  val uniqueSrcIds = uniqueSrcIdsBuilder.result()
  val numUniqueSrdIds = uniqueSrcIds.length
  val dstCounts = dstCountsBuilder.result()
  val dstPtrs = new Array[Int](numUniqueSrdIds + 1)
  var sum = 0
  //将src id和dst id关系通过dstPtrs进行压缩存储
  i = 0
  while (i < numUniqueSrdIds) {
    sum += dstCounts(i)
    i += 1
    dstPtrs(i) = sum
  }
  InBlock(uniqueSrcIds, dstPtrs, dstEncodedIndices, ratings)
}
```

### computeFactor

  根据srcFactorBlocks、srcOutBlocks、dstInBlocks, 计算dstFactorBlocks

    
```scala
private def computeFactors[ID](
    srcFactorBlocks: RDD[(Int, FactorBlock)],
    srcOutBlocks: RDD[(Int, OutBlock)],
    dstInBlocks: RDD[(Int, InBlock[ID])],
    rank: Int,
    regParam: Double,
    srcEncoder: LocalIndexEncoder,
    implicitPrefs: Boolean = fALSe,
    alpha: Double = 1.0,
    solver: LeastSquaresNESolver): RDD[(Int, FactorBlock)] = {
  val numSrcBlocks = srcFactorBlocks.partitions.length  //src block数量
  val YtY = if (implicitPrefs) Some(computeYtY(srcFactorBlocks, rank)) else None
  //根据srcOut，得到每个dstBlock对应的srcBlockID 和srcFactor数组
  val srcOut = srcOutBlocks.join(srcFactorBlocks).flatMap {
    case (srcBlockId, (srcOutBlock, srcFactors)) =>
      
      srcOutBlock.view.zipWithIndex.map { case (activeIndices, dstBlockId) =>
        (dstBlockId, (srcBlockId, activeIndices.map(idx => srcFactors(idx))))
      }
  }
  //根据dstBlockId 对srcBlockID, array[srcFactor]进行聚合
  val merged = srcOut.groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))
  //对每个dstBlockID, 计算其中每个dstID对应的隐语义向量
  dstInBlocks.join(merged).mapValues {
    case (InBlock(dstIds, srcPtrs, srcEncodedIndices, ratings), srcFactors) =>
      //得到每个block对应的src factor向量集合
val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      srcFactors.foreach { case (srcBlockId, factors) =>
        sortedSrcFactors(srcBlockId) = factors
      }
      //对每个dstID, 获取对应的srcFactor及对应rating, 计算该dstID对应的隐语义向量
      val dstFactors = new Array[Array[Float]](dstIds.length)
      var j = 0
      val ls = new NormalEquation(rank)
      while (j < dstIds.length) {
        ls.reset()
        if (implicitPrefs) {
          ls.merge(YtY.get)
        }
        var i = srcPtrs(j)
        var numExplicits = 0
        while (i < srcPtrs(j + 1)) { //依次得到每个srcFactor及rating值
          val encoded = srcEncodedIndices(i)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          //sortedSrcFactors通过blockId和localIndex进行索引，得到需要的factor向量。之前这里困惑挺久，一直感觉从srcOut传过来的factor向量只是一个子集，通过localIndex访问不正确，实际上这里的localIndex和srcOut那里存储的localindex是不需要对应的。因为同一个src id 本身的src local index不等于其它block对应的 dst localindex
          val srcFactor = sortedSrcFactors(blockId)(localIndex)
          val rating = ratings(i)
          if (implicitPrefs) {
            // Extension to the original paper to handle b < 0. confidence is a function of |b|
            // instead so that it is never negative. c1 is confidence - 1.0.
            val c1 = alpha * math.abs(rating)
            // For rating <= 0, the corresponding preference is 0. So the term below is only added
            // for rating > 0. Because YtY is already added, we need to adjust the scaling here.
            if (rating > 0) {
              numExplicits += 1
              ls.add(srcFactor, (c1 + 1.0) / c1, c1)
            }
          } else {
            ls.add(srcFactor, rating)
            numExplicits += 1
          }
          i += 1
        }
        // Weight lambda by the number of explicit ratings based on the ALS-WR paper.
        dstFactors(j) = solver.solve(ls, numExplicits * regParam)
        j += 1
      }
      dstFactors
  }
}
```


## ALS 模型推荐
    
**模型参数** 

```
 val rank: Int,      //隐语义个数
 val userFeatures: RDD[(Int, Array[Double])], //user factor数组, 存储user id 及对应的factor向量
 val productFeatures: RDD[(Int, Array[Double])]) //item factor数组，存储item id及对应的factor向量
    
```

**对所有用户进行推荐**

调用recommendForAll函数，首先对user向量和item向量分块并以矩阵形式存储，然后对二者做笛卡尔积，并计算每个user和每个item的得分，最终以user为key, 取topK个item及对应的得分，作为推荐结果. 计算topK时借助于小顶堆

```
  private def recommendForAll(
      rank: Int,
      srcFeatures: RDD[(Int, Array[Double])],
      dstFeatures: RDD[(Int, Array[Double])],
      num: Int): RDD[(Int, Array[(Int, Double)])] = {
    //对user向量和item向量分块并以矩阵形式存储
    val srcBlocks = blockify(rank, srcFeatures)
    val dstBlocks = blockify(rank, dstFeatures)
    //笛卡尔积，依次对每个组合计算user对item的偏好
    val ratings = srcBlocks.cartesian(dstBlocks).flatMap {
      case ((srcIds, srcFactors), (dstIds, dstFactors)) =>
        val m = srcIds.length
        val n = dstIds.length
        val ratings = srcFactors.transpose.multiply(dstFactors)
        val output = new Array[(Int, (Int, Double))](m * n)
        var k = 0
        ratings.foreachActive { (i, j, r) =>
          output(k) = (srcIds(i), (dstIds(j), r))
          k += 1
        }
        output.toSeq
    }
    //根据user id作为key, 得到喜好分数最高的num个item
    ratings.topByKey(num)(Ordering.by(_._2))
  }


  // 对user向量和item向量分块并以矩阵形式存储, 结果的每个元组分别是对应的id数组和factor构成的矩阵
  private def blockify(
      rank: Int,
      features: RDD[(Int, Array[Double])]): RDD[(Array[Int], DenseMatrix)] = {
    val blockSize = 4096 // TODO: tune the block size
    val blockStorage = rank * blockSize
    features.mapPartitions { iter =>
      iter.grouped(blockSize).map { grouped =>
        val ids = mutable.ArrayBuilder.make[Int]
        ids.sizeHint(blockSize)
        val factors = mutable.ArrayBuilder.make[Double]
        factors.sizeHint(blockStorage)
        var i = 0
        grouped.foreach { case (id, factor) =>
          ids += id
          factors ++= factor
          i += 1
        }
        (ids.result(), new DenseMatrix(rank, i, factors.result()))
      }
    }
  }

```



【1】Y Koren，R Bell，C Volinsky, "Matrix Factorization Techniques for Recommender Systems", 《Computer》, 2009.08; 42(8):30-37 

【2】洪亮劼, "知人知面需知心——人工智能技术在推荐系统中的应用", 2016.11, http://mp.weixin.qq.com/s/JuaM8d52-f8AzTjEPnCl7g

【3】S. Funk, “Netflix Update: Try This at Home”, 2006.12, http://sifter.org/~simon/journal/20061211.html

【4】Y. Koren, “Factorization Meets the Neighborhood: A Mul-tifaceted Collaborative Filtering Model”, Proc. 14th ACM SIGKDD Int’l Conf. Knowledge Discovery and Data Mining, ACM Press, 2008, pp.426-434

【5】A. Paterek, “Improving Regularized Singular Value De-composition for Collaborative Filtering” Proc. KDD Cup and Workshop, ACM Press, 2007, pp.39-42

【6】G. Takács et al., “Major Components of the Gravity Recom- mendation System”, SIGKDD Explorations, vol. 9, 2007, pp.80-84

【7】孟祥瑞, "ALS 在 Spark MLlib 中的实现", 2015.05, http://www.csdn.net/article/2015-05-07/2824641

