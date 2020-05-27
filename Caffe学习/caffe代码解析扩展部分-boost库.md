>BOOST库

# 1 编译验证
在`<boost/version.hpp> `中有两个宏，
```c++
#ifndef BOOST_VERSION_HPP
#define BOOST_VERSION_HPP

//
//  Caution: this is the only Boost header that is guaranteed
//  to change with every Boost release. Including this header
//  will cause a recompile every time a new Boost version is
//  used.
//
//  BOOST_VERSION % 100 is the patch level
//  BOOST_VERSION / 100 % 1000 is the minor version
//  BOOST_VERSION / 100000 is the major version

#define BOOST_VERSION 106600

//
//  BOOST_LIB_VERSION must be defined to be the same as BOOST_VERSION
//  but as a *string* in the form "x_y[_z]" where x is the major version
//  number, y is the minor version number, and z is the patch level if not 0.
//  This is used by <config/auto_link.hpp> to select which library version to link to.

#define BOOST_LIB_VERSION "1_66"

#endif
```

cmake代码：
```cmake
# CMakeLists.txt
project(Demo-Boost)
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 14)

set(BOOST_ROOT /tmp/tmp.dW9ZaTjkMM/boost_1_66_0)

find_package(Boost COMPONENTS regex system REQUIRED)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})

    MESSAGE( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
    MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
    MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")

    add_executable(demo main.cpp)
    target_link_libraries (demo ${Boost_LIBRARIES})
endif()

```
# 2 时间与日期库

## 2.1 timer库概述
```c++
#include <boost/config.hpp>
#include <boost/version.hpp>
#include <boost/timer.hpp>
#include <iostream>
using namespace std;
using namespace boost;

int main()
{
    cout << "BOOST_VERSION: " << BOOST_VERSION << endl;   //Boost版本号
    cout <<  "BOOST_LIB_VERSION: " << BOOST_LIB_VERSION << endl;
    timer t;    //声明一个时间对象,开始计时

    cout << "max timespan: " //可度量的最大时间 h
    << t.elapsed_max() / 3600 << "h" <<endl;

    cout<<"min timespan: "  //可度量的最小时间 s
    << t.elapsed_min()  << "s" << endl;

    cout<<"now time elapsed: "
    <<t.elapsed() <<"s"<<endl;

}
/*
BOOST_VERSION: 106600
BOOST_LIB_VERSION: 1_66
max timespan: 2.56205e+09h
min timespan: 1e-06s
now time elapsed: 5.3e-05s
*/
```

## 2.2 progress_timer

### 2.2.1 progress_timer 用法
```c++
#include <boost/config.hpp>
#include <boost/version.hpp>
#include <boost/timer.hpp>
#include <iostream>
#include <boost/progress.hpp>

using namespace std;
using namespace boost;

void test() {
    progress_timer t1;

}

int main()
{
    cout << "BOOST_VERSION: " << BOOST_VERSION << endl;   //Boost版本号
    cout << "BOOST_LIB_VERSION: " << BOOST_LIB_VERSION << endl;

    progress_timer t;   //声明一个 progress_timer 对象
    for (int i = 0; i < 100; i++)
        cout << "i: " << i;
    test();


}
```

### 2.2.2 progress_timer源码解析
C++继承：
1）公有继承——public：基类的公有变为派生类的公有，基类的保护变为派生类的保护，私有派生类不可访问
2）私有继承——private：基类的公有变为派生类的私有，基类的保护变为派生类的私有，基类的私有不可访问
3）保护继承——protect：基类的公有变为派生类的保护，基类的保护变为派生类的保护，基类的私有不可访问

**派生类不能继承基类的构造函数和析构函数**
![](MarkdownImg/2020-04-21-16-20-25.png)

explict 禁止隐式转换
>explicit关键字只需用于类内的单参数构造函数前面。由于无参数的构造函数和多参数的构造函数总是显示调用，这种情况在构造函数前加explicit无意义。
google的c++规范中提到explicit的优点是可以避免不合时宜的类型变换，缺点无。所以google约定所有单参数的构造函数都必须是显示的，只有极少数情况下拷贝构造函数可以不声明称explicit。例如作为其他类的透明包装器的类。
effective c++中说：被声明为explicit的构造函数通常比其non-explicit兄弟更受欢迎。因为它们禁止编译器执行非预期（往往也不被期望）的类型转换。除非我有一个好理由允许构造函数被用于隐式类型转换，否则我会把它声明为explicit，鼓励大家遵循相同的政策。




```c++
class progress_timer : public timer, private noncopyable
{
  
 public:
  explicit progress_timer( std::ostream & os = std::cout )
     // os is hint; implementation may ignore, particularly in embedded systems
     : timer(), noncopyable(), m_os(os) {}
  ~progress_timer()
  {
  //  A) Throwing an exception from a destructor is a Bad Thing.
  //  B) The progress_timer destructor does output which may throw.
  //  C) A progress_timer is usually not critical to the application.
  //  Therefore, wrap the I/O in a try block, catch and ignore all exceptions.
    try
    {
      // use istream instead of ios_base to workaround GNU problem (Greg Chicares)
      std::istream::fmtflags old_flags = m_os.setf( std::istream::fixed,
                                                   std::istream::floatfield );
      std::streamsize old_prec = m_os.precision( 2 );
      m_os << elapsed() << " s\n" // "s" is System International d'Unites std
                        << std::endl;
      m_os.flags( old_flags );
      m_os.precision( old_prec );
    }

    catch (...) {} // eat any exceptions
  } // ~progress_timer

 private:
  std::ostream & m_os;
};
```
## 2.3 progress_display
```c++
#include <fstream>
ofstream     //文件写操作,内存写入存储设备(文件)  输出流
//通常我们所说的对一个文件进行写操作，就是把内存里的内容，也就是缓冲区的内容写到硬盘，可以将标准输出设备理解为显示器
ifstream      //文件读操作,存储设备到内存.       输入流
//通常我们所说对一个文件读操作，就是把存在硬盘的内容写到内存中，也就是缓冲区
fstream      //读写操作,对打开的文件可进行读写.   前两者的结合
```

```c++
#include <boost/version.hpp>
#include <iostream>
#include <boost/progress.hpp>
#include <cstring>
#include <vector>
#include <fstream>

using namespace std;
using namespace boost;

int main()
{
    cout << "BOOST_VERSION: " << BOOST_VERSION << endl;   //Boost版本号
    cout << "BOOST_LIB_VERSION: " << BOOST_LIB_VERSION << endl;
    vector<string> v(100);  //一个字符串向量
    ofstream fs("./test,txt");  //文件输出流
    cout << "v.size(): " << v.size();
    progress_display pd(v.size());  //声明一个 progress_display 对象，基数是v的大小

    for (auto& x:v) {   //for+auto循环
        fs << x << endl;
        ++pd;   //更新进度显示
    }
}
```

```c++
#include <boost/version.hpp>
#include <iostream>
#include <boost/progress.hpp>
#include <cstring>
#include <vector>
#include <fstream>

using namespace std;
using namespace boost;

int main()
{
    cout << "BOOST_VERSION: " << BOOST_VERSION << endl;   //Boost版本号
    cout << "BOOST_LIB_VERSION: " << BOOST_LIB_VERSION << endl;
    vector<string> v(100,"aa");  //一个字符串向量
    v[10]="";
    v[23] = "";
    ofstream fs("../test.txt");  //文件输出流
    cout << "v.size(): " << v.size();
    progress_display pd(v.size());  //声明一个 progress_display 对象，基数是v的大小

    for (auto pos = v.begin(); pos != v.end(); ++pos) {
        fs << *pos << endl;
        pd.restart(v.size());
//        ++pd;
        pd += (pos - v.begin() + 1);
        if(pos->empty()) {
            cout << "null string # "
                 << (pos - v.begin()+1) << endl;
        }
    }
}
```
## 2.4 datetime库

```c++
//处理日期的组件
#include <boost/date_time/gregorian/gregorian.hpp>
using namespace boost::gregorian;
//处理时间的组件
#include <boost/date_time/posix_time/posix_time.hpp>
using namespace boost::posix_time;

int main()
{
    date d1;
    date d2(2010, 1, 1);
    date d3(2000, Jan, 1);
    date d4(d2);

    assert(d1 == date(not_a_date_time));
    assert(d4 == d2);
    assert(d3 < d4);

    cout << day_clock::local_day() << endl; //本地时间
    cout << day_clock::universal_day() << endl; //UTC日期
}
```
# 3 内存管理

## 3.1 smart_ptr

### 3.1.1 RAII机制
RAII机制，资源获取即初始化，(Resource Acquisition IS Initialization),在类的构造函数里申请资源，然后使用，在析构函数中释放资源。
boost.smart_ptr库提供了六种智能指针：scoped_ptr,scoped_array,shared_ptr,shared_array,weak_ptr和intrusive_ptr.

## 3.2 scoped_ptr

```c++
template<class T> class scoped_ptr // noncopyable
{
private:

    T * px; //原始指针

    scoped_ptr(scoped_ptr const &); //拷贝构造函数私有化
    scoped_ptr & operator=(scoped_ptr const &); //赋值操作私有化

    typedef scoped_ptr<T> this_type;

    void operator==( scoped_ptr const& ) const; //相等操作私有化
    void operator!=( scoped_ptr const& ) const; //不等操作私有化

public:

    typedef T element_type; 

    explicit scoped_ptr( T * p = 0 ) BOOST_SP_NOEXCEPT : px( p )
    {
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
        boost::sp_scalar_constructor_hook( px );
#endif
    }   //显示构造函数

#ifndef BOOST_NO_AUTO_PTR

    explicit scoped_ptr( std::auto_ptr<T> p ) BOOST_SP_NOEXCEPT : px( p.release() )
    {
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
        boost::sp_scalar_constructor_hook( px );
#endif
    }

#endif

    ~scoped_ptr() BOOST_SP_NOEXCEPT
    {
#if defined(BOOST_SP_ENABLE_DEBUG_HOOKS)
        boost::sp_scalar_destructor_hook( px );
#endif
        boost::checked_delete( px );
    }   //析构函数

    void reset(T * p = 0) BOOST_SP_NOEXCEPT_WITH_ASSERT
    {
        BOOST_ASSERT( p == 0 || p != px ); // catch self-reset errors
        this_type(p).swap(*this);
    }   //重置智能指针

    T & operator*() const BOOST_SP_NOEXCEPT_WITH_ASSERT
    {
        BOOST_ASSERT( px != 0 );
        return *px;
    }   //操作符重载

    T * operator->() const BOOST_SP_NOEXCEPT_WITH_ASSERT
    {
        BOOST_ASSERT( px != 0 );
        return px;
    }   //操作符重载

    T * get() const BOOST_SP_NOEXCEPT
    {
        return px;
    }   //获取原始指针

// implicit conversion to "bool"
#include <boost/smart_ptr/detail/operator_bool.hpp>

    void swap(scoped_ptr & b) BOOST_SP_NOEXCEPT
    {
        T * tmp = b.px;
        b.px = px;
        px = tmp;
    }
};  //交换指针

```
scoped_ptr 的构造函数接受一个类型为 T* 的指针p,创建出一个 scoped_ptr对象，scoped_ptr同时把拷贝构造函数和赋值操作符都声明为私有的，静止对智能指针的拷贝操作。保证了被他管理的指针不能被转让所有权。

```c++
#include <iostream>
#include <boost/smart_ptr.hpp>

using namespace std;
using namespace boost;

struct posix_file {
    posix_file(const char *file_name) {
        cout << "opend file: " << file_name << endl;
    }

    virtual ~posix_file() {
        cout << "close file: " << endl;
    }
};

int main()
{
    scoped_ptr<posix_file> fp(new posix_file("../a.txt"));
    scoped_ptr<int> p(new int);

    if (p) {
        *p = 100;
        cout << *p << endl;
    }
    p.reset();

    assert(p==0);
    if (!p) {
        cout << "scoped_ptr == nullptr" << endl;
    }

}
// opend file: ../a.txt
// 100
// scoped_ptr == nullptr
// close file: 
```

###  3.2.1 unique_ptr
c++ 标准中对 unique_ptr 的基本能力与 scoped_ptr相同，同样可以用于在作用域内管理指针，也不允许拷贝构造和拷贝赋值。

```c++
#include <iostream>
#include <boost/smart_ptr.hpp>

using namespace std;
using namespace boost;



int main()
{
    unique_ptr<int> up(new int);    //声明一个unique_ptr,管理int指针
    assert(up); //bool语境测试指针是否有效
    *up = 10;   //使用 operator* 操作指针
    cout << *up << endl;

    up.reset(); //释放指针
    assert(!up);    //此时不管理任何指针

}
```

## 3.3 shared_ptr
shared_ptr实现的是引用计数型的智能指针，可以被自由拷贝和赋值，在任意的地方共享它，当没有代码使用(引用计数为0)它才删除被包装的动态分配的对象。

shared_ptr也可以安全地放入标准容器中。

### 3.3.1 操作函数
shared_ptr与scoped_ptr同样是用于管理new动态分配对象的智能指针，因此功能上有很多相似之处，不能管理 new[]产生的动态数组指针。
列如：
```c++
#include <iostream>
#include <boost/smart_ptr.hpp>

using std::cout;
using std::endl;
using std::string;
using namespace boost;

int main()
{
    shared_ptr<int> spi(new int);   //一个int的shared_ptr
    assert(spi);    //在bool语境中转换为 bool
    *spi = 253;     //使用解引用操作符 *

    shared_ptr<string> sps(new string("smart"));    //一个string的ptr
    assert(sps->size()==5);     //使用箭头操作符 ->

    shared_ptr<int> dont_do_this(new int[10]);  //危险！不能正确释放内存
    
}
```

**typedef的用法**
```c++
int main()
{
    typedef int NUM;
//    NUM p(10);
//    NUM p=10;
    NUM(p) = 10;
    cout << p << endl;
}




```







**map的使用**

```c++

```















































