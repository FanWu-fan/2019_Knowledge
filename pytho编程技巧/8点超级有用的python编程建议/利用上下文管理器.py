class raise_api_error:
    """
    capture specified exception and raise ApiErrorCode instead 
    "raises: AttributeError if code_name is not valid
    """
    def __init__(self,captures,code_name):
        self.captures = captures
        self.code = getattr(error_codes,code_name)
    
    def __enter__(self):
        #该方法在进入上下文时调用
        return self
    
    def __exit__(self,exc_type,exc_val,exv_tb):
        #该方法在退出上下文时调用，
        #exc_type,exc_val,exc_tb分别表示在上下文内抛出
        #异常类型，异常值，错误栈
        if exc_type is None:
            return False
        
        if exc_type == self.captures:
            raise self.code from exc_val
        return False


def upload_avatar(request):
   """用户上传新头像"""
with raise_api_error(KeyError, 'AVATAR_FILE_NOT_PROVIDED'):
   avatar_file = request.FILES['avatar']

with raise_api_error(ResizeAvatarError, 'AVATAR_FILE_INVALID'),
       raise_api_error(FileTooLargeError, 'AVATAR_FILE_TOO_LARGE'):
     resized_avatar_file = resize_avatar(avatar_file)

with raise_api_error(Exception, 'INTERNAL_SERVER_ERROR'):
   request.user.avatar = resized_avatar_file
   request.user.save()
return HttpResponse({})