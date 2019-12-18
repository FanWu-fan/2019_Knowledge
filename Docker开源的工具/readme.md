> 摘抄原网址： https://mp.weixin.qq.com/s?__biz=MjM5MDE0Mjc4MA==&mid=2651023037&idx=2&sn=ddbb8c2b882e0fe58c4bd2930b5ade65&chksm=bdbe9eee8ac917f8c434d9804199c1afd5df36b6dd863c96b9d13c7a6ad4a4d14cf088262f8f&mpshare=1&scene=1&srcid=&sharer_sharetime=1576543227678&sharer_shareid=c6dddb77371c457f894e87ca845a013d&key=185314f0b5c82f073b9bcb1bde065d75c2bf5c0721f8776e50cc894ce4e5293605c25ee17846dce8c47b61d032525a107730ccc4958fc95eb7ce89acbdbe94d0fdb98244c6c5266b835c6f2894622a60&ascene=1&uin=MTM2MTEwNTY4NA%3D%3D&devicetype=Windows+10&version=62070158&lang=zh_CN&exportkey=AZAGQNuoTqmJgoiU0vHS8FY%3D&pass_ticket=G2xApLhu0MhZemb4KHh8C3UX6RzMP4L1yqadu%2FXPctoV7E7UrOIlXtups9V4ew6T

# Docker开源工具推荐

## docker-slim（容器瘦身）
> https://github.com/docker-slim/docker-slim
> https://github.com/docker-slim/docker-slim/blob/master/README.md

docker-slim通过各种分析技术了解你的应用程序的需求，并且优化和保护你的容器。它将抛弃你容器内不需要的资源，减小你容器受攻击的点。如果你需要一些额外的东西来调试你的容器，你可以使用（ dedicated debugging）专用调试器

docker-slim 已经被用在 *Node.js, Python, Ruby, Java, Golang, Rust, Elixir and PHP (some app types)* 

当docker-slim运行时，它使您有机会与它创建的零时容器进行交互。默认情况下，在继续执行之前，它将暂停并等待您的输入。你可以使用 *continue-after*标志更改此行为。

如果您的应用程序暴露任何web接口(列如，当您拥有web服务器或HTTP API时)，您将看到主机上与应用程序交互所需的端口号（在 port.list 和 target.port.info在屏幕上），比如，在屏幕广播中，您将看到内部应用程序端口8000 被映射到 主机上的端口 32911

请注意，如果您使用 HTTP probing(仔细观察) --http-probe标志启用 HTTP探测，一些使用脚本语言（ru python或Ruby）需要服务交互来加载应用程序中的所有内容，启用HTTP探测，除非它妨碍了你。