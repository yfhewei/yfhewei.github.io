# 断约的预测 / Prediction of Category Customer Churn
## 问题及背景 / Background
* On a multi-category commodity procurement platform, customers may stop purchasing certain goods for various reasons after they have made a purchase. We refer to this phenomenon as “Category Customer Churn”.
* Category Customer Churn can reduce the sales volume of the platform's goods. If Category Customer Churn is detected early, remedial operational measures can be taken in advance, such as offering alternative products. However, if the judgment of customer-product defection is incorrect, offering substitutes may disturb the customer's normal purchasing and even have a negative impact. Therefore, it is necessary to timely and accurately identify the Category Customer Churn, and provide suitable alternatives.
* 在一个多种类商品采购平台上，对已购买过的商品，客户可能会因为种种原因不再购买。这种现象我们称之为客-品断约。
* 客-品断约会降低平台商品的销量。如果能尽早发现客-品断约，则可以提前采取运营措施补救，如提供替代商品。如果对客-品断约的判断是错误的，则提供替代品会对客户的正常购买造成困扰，甚至带来负面影响。因此，需要及时准确的找出断约的客-品对，并提供合适的替代品。
## 问题定义 / Define the problem
* From business perspective, customer-category that have been transacted within M days are considered the target group to focus on. The definition of Category Customer Churn among them is that if there are no transactions within N days after the last transaction for Category Customer, it is considered a Category Customer Churn. This approach uses a unified standard of a N-day window period to determine the occurrence of Churn. In practice, the timeliness and accuracy of Churn detection can be balanced by adjusting the size of N.
* This definition does not take into account the differences in purchase cycles between different types of goods, nor does it consider the differences in purchase cycles among different customers. Therefore, for different customer category, some may be detected too late, while others may be mistakenly judged too early.
* 从业务角度，将M天内成交过的客户品类，看作是需要关注的目标群体。对他们中的断约客-品的定义，是最后一次成交后N天内再无成交的客-品，看作是发生了断约。这种做法用统一的标准N天空窗期，来判断断约的发生。在实际中可以通过调整N的大小来平衡断约发现的及时性和准确性。
* 这一定义，没有考虑不同种类商品之间的购买周期差异，也没有考虑不同客户之间的购买周期差异。因此对于不同客-品，会出现有的发现的过晚，而有些因为太早而做出了误判。
## 数据分析
* 一般这种电商平台可以获取的数据是客户在平台的浏览行为日志数据，以及客户的下单历史记录数据。
* 通过对客户历史购买行为数据进行分析，可以考察客户的购买是否为首次购买、客户在购买之后是否持续有浏览行为、客户是否会切换购买相似品等因素，分析这些因素与客户复购行为之间的关联关系。
## 方案设计
* 训练：对每日的下单客-品，构建N个独立的分类预测模型，分别预测客户在下单后，1～N-1天未下单的假设条件下，后续剩余天内下单概率。这是一个二分类模型，训练数据取对应条件下的历史数据，特征则考虑客户历史行为、商品特性及客品交互关系进行构建。
* 预测：每天拿到客户下单数据后，利用训练好的模型对各种前提条件下客户的后续下单概率分别进行预测，N-1个模型得到N-1个预测结果，保存在数据表中。如果该客-品有历史预测结果，则用新的预测结果更新替换历史预测数据。
* 应用，对未下单的待预测目标客-品，计算其当天未下单距离其最后一次下单的时间间隔，根据这个数值选择对应的预测结果，然后再根据预测结果的概率值，以及预设的阈值，决定是否预测其为断约客品。

