#  0 Caffe依赖包解析

# 1. ProtoBuffer
## 1.1 ProtoBuffer
`ProtoBuffer`是由Google开发的一种可以实现内存和非易失性存储介质(如硬盘文件)交换的协议接口。Caffe源码中大量使用`ProtoBuffer`作为权值和模型参数的载体。一般开发者有的喜欢TXT的易于修改，有的喜欢BIN的读写高效，一个项目组内不同成员必须约定一套统一的参数方案。`ProtoBuffer`工具完美地解决了这个问题，**用户只需要建立统一的参数描述文件(proto)**,然后利用 protoc 编译就能让协议细节等关键部分代码自动生成，节省开发时间。使用 `ProtoBuffer` 还可以跨语言(C++/Java/Python)传递相同的数据结构，让团队协作更有效率。

## 1.2 为何使用Protocol Buffers?
我们将要使用的示例是一个非常简单的 “地址簿” 应用程序，可以在文件中读写联系人的详细信息。地址簿中的每个人都有姓名、ID、电子邮件地址和联系电话。

你该如何序列化和反序列化如上结构的数据呢？这里有几种解决方案：

* 可以以二进制形式发送/保存原始内存中数据结构。随着时间的推移，这是一种脆弱的方法，因为接收/读取代码必须使用完全相同的内存布局、字节顺序等进行编译。此外，由于文件以原始格式累积数据，并且解析该格式的软件副本四处传播，因此很难扩展格式。

* 你可以发明一种特殊的方法将数据项编码为单个字符串 - 例如将 4 个整数编码为 "12:3:-23:67"。这是一种简单而灵活的方法，虽然它确实需要编写一次性编码和解析的代码，并且解析会产生一些小的运行时成本。但这非常适合非常简单的数据的编码。


* 将数据序列化为 XML。这种方法非常有吸引力，因为 XML（差不多）是人类可读的，并且有许多语言的绑定库。如果你想与其他应用程序/项目共享数据，这可能是一个不错的选择。然而，XML 是众所周知需要更多的空间，并且编码/解码 XML 会对应用程序造成巨大的性能损失。此外，导航 XML DOM 树比通常在类中导航简单字段要复杂得多。

而 Protocol buffers 是灵活，高效，自动化的解决方案。采用 protocol buffers，你可以写一个 `.proto` 文件描述你想要读取的数据的结构。由此， protocol buffer 编译器将创建一个类，该类使用有效的二进制格式实现 protocol buffer 数据的自动编码和解析。生成的类为构成 protocol buffer 的字段提供 getter 和 setter，并负责读写 protocol buffer 单元的细节。重要的是，protocol buffer 的格式支持随着时间的推移扩展格式的想法，使得代码仍然可以读取用旧格式编码的数据。

## 1.3 定义你的 protol 格式
要创建地址簿应用程序，你需要从 .proto 文件开始。.proto 文件中的定义很简单：为要序列化的每个数据结构添加 message 定义，然后为 message 中的每个字段指定名称和类型。下面就是定义相关 message 的 `.proto` 文件，addressbook.proto。
```proto
syntax = "proto2";

package tutorial;

message Person {
  required string name = 1;
  required int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    required string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

`.proto` 文件以 `package` 声明开头，这有助于防止不同项目之间的命名冲突。在 C++ 中，生成的类将放在与包名匹配的 `namespace` （命名空间）中。

接下来，你将看到相关的 `message` 定义。 `message` 只是包含一组类型字段的集合。许多标准的简单数据类型都可用作字段类型，包括 bool、int32、float、double 和 string。你还可以使用其他 message 类型作为字段类型在消息中添加更多结构 - 在上面的示例中， `Person`  包含  `PhoneNumber message`  ，而  `AddressBook`  包含  `Person message` 。你甚至可以定义嵌套在其他 message 中的 message 类型 -​​ 如你所见， `PhoneNumber`  类型在  `Person`  中定义。如果你希望其中一个字段具有预定义的值列表中的值，你还可以定义 `枚举类型`  - 此处你指定（枚举）电话号码，它的值可以是 MOBILE，HOME 或 WORK 之一。

每个元素上的 "=1"，"=2" 标记表示该字段在二进制编码中使用的唯一 “标记”。标签号 1-15 比起更大数字需要少一个字节进行编码，因此以此进行优化，你可以决定将这些标签用于常用或重复的元素，将标记 16 和更高的标记留给不太常用的可选元素。 `repeated`  字段中的每个元素都需要重新编码  `Tag` ，因此 repeated 字段特别适合使用此优化。

必须使用以下修饰符之一注释每个字段：
* **required** : 必须提供该字段的值，否则该消息将被视为“未初始化”。如果是在调试模式下编译 libprotobuf，则序列化一个未初始化的 message 将将导致断言失败。在优化的构建中，将跳过检查并始终写入消息。但是，解析未初始化的消息将始终失败（通过从解析方法返回 false）。除此之外，required 字段的行为与 optional 字段完全相同。

* **optional**: 可以设置也可以不设置该字段。如果未设置可选字段值，则使用默认值。对于简单类型，你可以指定自己的默认值，就像我们在示例中为电话号码类型所做的那样。否则，使用系统默认值：数字类型为 0，字符串为空字符串，bools 为 false。对于嵌入 message，默认值始终是消息的 “默认实例” 或 “原型”，其中没有设置任何字段。调用访问器以获取尚未显式设置的 optional（或 required）字段的值始终返回该字段的默认值。


* **repeated**: 该字段可以重复任意次数（包括零次）。重复值的顺序将保留在 protocol buffer 中。可以将 repeated 字段视为动态大小的数组。

## 1.4 编译你的 Protocol Buffers
既然你已经有了一个 **.proto** 文件，那么你需要做的下一件事就是生成你需要读写 **AddressBook** （以及 **Person** 和 **PhoneNumber** ） message 所需的类。为此，你需要在 .proto 上运行 protocol buffer 编译器 protoc：
`protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto`

因为你需要 C ++ 类，所以使用 --cpp_out 选项 - 当然，为其他支持的语言也提供了类似的选项。

这将在指定的目标目录中生成以下文件：
* `addressbook.pb.h`： 类声明的头文件
* `addressbook.pb.cc`：类实现

## 1.5 The Protocol Buffer API
让我们看看一些生成的代码，看看编译器为你创建了哪些类和函数。如果你查看　`addressbook.pb.h`，你会发现你在 `addressbook.proto` 中指定的每条 message 都有一个对应的类。仔细观察 Person 类，你可以看到编译器已为每个字段生成了访问器。例如，对于 `name` ， `id` ， `email` 和  `phone`  字段，你可以使用以下方法：
```
 // required name
  inline bool has_name() const;
  inline void clear_name();
  inline const ::std::string& name() const;
  inline void set_name(const ::std::string& value);
  inline void set_name(const char* value);
  inline ::std::string* mutable_name();

  // required id
  inline bool has_id() const;
  inline void clear_id();
  inline int32_t id() const;
  inline void set_id(int32_t value);

  // optional email
  inline bool has_email() const;
  inline void clear_email();
  inline const ::std::string& email() const;
  inline void set_email(const ::std::string& value);
  inline void set_email(const char* value);
  inline ::std::string* mutable_email();

  // repeated phones
  inline int phones_size() const;
  inline void clear_phones();
  inline const ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >& phones() const;
  inline ::google::protobuf::RepeatedPtrField< ::tutorial::Person_PhoneNumber >* mutable_phones();
  inline const ::tutorial::Person_PhoneNumber& phones(int index) const;
  inline ::tutorial::Person_PhoneNumber* mutable_phones(int index);
  inline ::tutorial::Person_PhoneNumber* add_phones();
  ```

如你所见，getter 的名称与小写字段完全相同，setter 方法以 set_ 开头。每个单数（required 或 optional）字段也有 has_ 方法，如果设置了该字段，则返回 true。最后，每个字段都有一个 clear_ 方法，可以将字段重新设置回 empty 状态。

虽然数字 id 字段只有上面描述的基本访问器集，但是 name 和 email 字段因为是字符串所以有几个额外的方法：一个 mutable_ 的 getter，它允许你获得一个指向字符串的直接指针，以及一个额外的 setter。请注意，即使尚未设置 email ，也可以调用 mutable_email()；它将自动初始化为空字符串。如果在这个例子中你有一个单数的 message 字段，它也会有一个 mutable_ 方法而不是 set_ 方法。

`repeated` 字段也有一些特殊的方法 - 如果你看一下 `repeated phones` 字段的相关方法，你会发现你可以：
* 检查 `repeated` 字段长度（换句话说，与此人关联的电话号码数）
* 使用索引获取指定的电话号码
* 更新指定索引处的现有电话号码
* 在 message 中添加另一个电话号码同时之后也可进行再修改（repeated 的标量类型有一个 add_，而且只允许你传入新值）

## 1.6 枚举和嵌套类
生成的代码包含与你的 `.proto `枚举对应的 `PhoneType` 枚举。你可以将此类型称为 `Person::PhoneType`，其值为 `Person::MOBILE`，`Person::HOME` 和 `Person::WORK`（实现细节稍微复杂一些，但你如果仅仅只是使用不需要理解里面的实现原理）。

编译器还为你生成了一个名为` Person::PhoneNumber` 的嵌套类。如果查看代码，可以看到 “真实” 类实际上称为 `Person_PhoneNumber`，但在 `Person` 中定义的 `typedef` 允许你将其视为嵌套类。唯一会造成一点差异的情况是，如果你想在另一个文件中前向声明该类 - 你不能在 C ++ 中前向声明嵌套类型，但你可以前向声明 `Person_PhoneNumber` 。

## 1.7 标准 Message 方法
每个 message 类还包含许多其他方法，可用于检查或操作整个 message，包括：

* `bool IsInitialized() const`;: 检查是否已设置所有必填 required 字段
* `string DebugString() const`;: 返回 message 的人类可读表达，对调试特别有用
* `void CopyFrom(const Person& from)`;: 用给定的 message 的值覆盖 message
* `void Clear()`;: 将所有元素清除回 empty 状态

这些和下一节中描述的 I/O 方法实现了所有 C++ protocol buffer 类共享的 Message 接口。

## 1.8 解析和序列化
最后，每个 protocol buffer 类都有使用 protocol buffer 二进制格式 读写所选类型 message 的方法。包括：

* bool SerializeToString(string* output) const;:序列化消息并将字节存储在给定的字符串中。请注意，字节是二进制的，而不是文本;我们只使用 string 类作为方便的容器。
* bool ParseFromString(const string& data);: 解析给定字符串到 message
* bool SerializeToOstream(ostream* output) const;: 将 message 写入给定的 C++ 的 ostream
* bool ParseFromIstream(istream* input);: 解析给定 C++ istream 到 message

## 1.9 caffe 内部使用
在caffe目录 `caffe/models/bvlc_reference_caffenet/solver.prototxt`中可以看到

```c++
net: "models/bvlc_reference_caffenet/train_val.prototxt"
test_iter: 1000
test_interval: 1000
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
stepsize: 100000
display: 20
max_iter: 450000
momentum: 0.9
weight_decay: 0.0005
snapshot: 10000
snapshot_prefix: "models/bvlc_reference_caffenet/caffenet_train"
solver_mode: GPU
```
这里记录了模型训练的超参数 (Hyper-Parameter),用caffe训练的时候会先读取这个文件，获取其中特定字段的数值，并根据此设置内存中模型训练时的超参数变量值，从文件中读取到内存的过程就是由 ProtoBUffer 工具协助完成的。

下面写一个简单的测试程序

在 `caffe/build/include/caffe/proto`中有ProBuffer相关API的文件`caffe_pb2.py  caffe.pb.cc  caffe.pb.h`
其中 `caffe.pb.cc  caffe.pb.h`是用于解析Caffe参数配置文件、将模型权值序列化/反序列化到磁盘的协议接口，我们编写测试程序如下：

# 2. BLAS
Caffe 中调用了 BLAS(Basic Linear Algebra Subprograms, 基本线性代数子程序)，最常用的 BLAS实现有 Intel MKL,ATLAS,OpenBLAS等，下面是最常用的两个函数，位于 `include/caffe/util/mathfunction.hpp`:
```c++
// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);
```
gemm 表示矩阵乘法，实现操作为： $C = alpha * op(A) * op(B) + beta * C$,其中 `CBLS_TRANSPOSE` 是一个枚举常量:
```
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
```
`TransA`可以实现四种 OP：{$A，A^T,共轭转置矩阵A^H，共轭矩阵 A^-$}
M,N,K为矩阵维度的信息，





