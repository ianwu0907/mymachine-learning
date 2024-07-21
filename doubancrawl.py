# _*_ coding=utf-8 _*_
from bs4 import BeautifulSoup
import re
import urllib.request,urllib.error
import xlwt
import sqlite3
import requests

findlink=re.compile(r'<a href="(.*?)">')  # 创建正则表达式对象，表示规则（字符串的模式）
findimgsrc=re.compile(r'<img.*src="(.*?)"',re.S)  # re.S让换行符包含在字符中
findtitle=re.compile(r'<span class="title">(.*)</span>')
findother=re.compile(r'<span class="other">(.*)</span>')
findrating=re.compile(r'<span class="rating_num" property="v:average">(.*)</span>')
findjudge=re.compile(r'<span>(\d*)人评价</span>')
findinq=re.compile(r'<span class="inq">(.*)</span>')
findbd=re.compile(r'<p class="">(.*?)</p>',re.S)
def main():
    # 豆瓣Top250网址(末尾的参数 ?start= 加不加都可以访问到第一页)
    baseUrl = "https://movie.douban.com/top250?start="
    # 1. 爬取网页并解析数据
    dataList = getData(baseUrl)
    # 2. 保存数据（以Excel形式保存）
    savePath = ".\\豆瓣电影Top250.xls"
    saveData(dataList, savePath)


# 得到一个指定URL的网页内容
def askURL(url):
    # 模拟头部信息，像douban服务器发送消息
    # User-Agent 表明这是一个浏览器（这个来自谷歌浏览器F12里 Network中 Request headers的User-Agent）
    # 每个人的用户代理可能不相同，用自己的就好。不要复制错了，否则会报418状态码。
    head = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
    }
    # 封装头部信息
    request = urllib.request.Request(url, headers=head)
    html = ""
    try:
        # 发起请求并获取响应
        response = requests.get(url, headers=head)
        response.raise_for_status()  # 如果响应状态码不是200，会引发HTTPError异常
        # 读取整个HTML源码，并解码为UTF-8
        html = response.content.decode("utf-8")
    except requests.HTTPError as e:
        # 异常捕获
        print("状态码：", e.response.status_code)
        print("原因：", e.response.reason)
    except requests.RequestException as e:
        print("请求错误：", e)
    return html

# 爬取网页，返回数据列表
def getData(baseurl):
    dataList = []
    # 爬取所有网页并获取需要的HTML源码
    for i in range(0, 10):  # doubanTop250 共有10页，每页25条。 range(0,10)的范围是[0,10)。
        url = baseurl + str(i * 25)  # 最终 url = https://movie.douban.com/top250?start=225
        html = askURL(url)  # 将每个页面HTML源码获取出来
        # 对页面源码逐一解析
        soup = BeautifulSoup(html, "html.parser")
        for item in soup.find_all('div', class_="item"):
            data = []
            item = str(item)
            link = re.findall(findlink, item)
            data.append(link)

            img=re.findall(findimgsrc,item)

            title = re.findall(findtitle, item)
            if len(title) == 2:
                ctitle = title[0]
                data.append(ctitle)
                otitle = title[1].replace("/", "")
                data.append(otitle)
            else:
                data.append(title[0])
                data.append('')

            other = re.findall(findother, item)[0]
            other =other.replace('/','', 1)
            other=re.sub('()',"",other)
            data.append(other)

            rating = re.findall(findrating, item)[0]
            data.append(rating)

            judge = re.findall(findjudge, item)[0]
            data.append(judge)

            inq = re.findall(findinq, item)
            if len(inq) != 0:
                inq = inq[0].replace("。", "")
                data.append(inq)
            else:
                data.append("")

            bd = re.findall(findbd, item)[0]
            bd = re.sub( "\\xa0|(<br/>)|<br|/\.\.\.|/\n"," ", bd)
            data.append(bd.strip())
            dataList.append(data)
    # 返回解析好的数据
    return dataList


# 保存数据
def saveData(dataList, savePath):
    print("正在保存...")

    book = xlwt.Workbook(encoding="utf-8")
    sheet = book.add_sheet("豆瓣电影Top250", cell_overwrite_ok=True)  # 第二个参数表示 可以对单元格进行覆盖

    # 写入列名（第一行存储列名）
    col = (
    "影片详情链接", "图片链接", "影片中文名", "影片外国名", "影片别称", "评分", "评论人数", "影片概括", "影片相关信息")
    for i in range(0, len(col)):
        sheet.write(0, i, col[i])

    # 写入电影信息（从第二行开始）
    for i in range(0, 250):
        print("正在保存->第%d部电影" % (i + 1))
        data = dataList[i]  # 取出某部电影数据
        for j in range(0, len(col)):
            if j < len(data):
                sheet.write(i + 1, j, data[j])
            else:
                sheet.write(i + 1, j, "")  # 如果data中没有足够的元素，写入空字符串

    book.save(savePath)  # 保存Excel

    print("保存完成!!")



# 程序入口
if __name__ == "__main__":
    # 调用函数
    main()
