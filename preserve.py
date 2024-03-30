import os
import time
import oss2
import argparse
from datetime import datetime


def list_files(bucket, remote_dir):
    for obj in oss2.ObjectIterator(bucket, prefix=remote_dir):
        print(obj.key)

def upload_file(bucket, local_file, remote_dir):
    if not os.path.isfile(local_file):
        print(f"Error: {local_file} does not exist or is not a file.")
        return
    new_local_filename = f'{os.path.basename(local_file)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.rename(local_file, new_local_filename)
    
    remote_upload_path = f'{remote_dir}/{os.path.basename(new_local_filename)}'
    bucket.put_object_from_file(remote_upload_path, new_local_filename)
    
    # 删除已上传的本地文件（可选，根据实际需求决定是否保留）
    os.remove(new_local_filename)

def download_file(bucket, remote_file, save_as):
    if not save_as:
        print("Error: Please provide a local file name to save the downloaded file.")
        return
    bucket.get_object_to_file(remote_file, save_as)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload/download/list files on Aliyun OSS")
    parser.add_argument("action", choices=["push", "get", "list"], help="The action to perform")
    parser.add_argument("-f", "--file", default="model.pth", help="Local file to push (default: model.pth)")
    parser.add_argument("-d", "--dir", default="mattingModel", help="Remote directory in OSS (default: mattingModel)")
    parser.add_argument("-s", "--save-as", help="Local file name to save when getting from OSS")

    args = parser.parse_args()

    auth = oss2.Auth('LTAI5tDgm7nf3f1tyvUwak3C', 'ae6DcwjznzILMSxlINSSH9SULdeZKy')
    endpoint = 'http://oss-cn-beijing.aliyuncs.com'
    bucket = oss2.Bucket(auth, endpoint, 'tdcasual')

    if args.action == "push":
        upload_file(bucket, args.file, args.dir)
    elif args.action == "get":
        download_file(bucket, f"{args.dir}/{args.file}", args.save_as)
    elif args.action == "list":
        list_files(bucket, args.dir)
    else:
        print("Invalid action specified.")