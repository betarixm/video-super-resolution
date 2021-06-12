from pixabay import Image, Video
import os
import requests
import time
import random


API_KEY = '21761960-2875c0e47a8fc9d8cf601ac8f'
save_root = './videos'

# video operations
video = Video(API_KEY)

# default video search
video.search()


total = 681 
keywords = ['rain']
for keyword in keywords:
    print(keyword)
    for ipage in range(1, 2):
        try:
            ipb_json = video.search(q=keyword, video_type='film',
                                    per_page=30,
                                    page=ipage)
        except Exception as e:
            print('Walking Error')
            continue

        videos = ipb_json['hits']
        for video in videos:
            duration = video['duration']
            if (duration > 20 or duration < 10):
                continue
            url = video['videos']['large']['url']
          
            if (str(url) != "None"):
                print(url)
                try:
                    response = requests.get(url)
                    ipath = os.path.join(save_root,
                                        str(total).zfill(3))
		with open(ipath, 'wb+') as f:
                        f.write(response.content)
                    total = total + 1;

                except Exception as e:
                    print(str(e))
                    continue


