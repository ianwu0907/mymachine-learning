import re
import requests
import pandas as pd

def get_url(cmd,page):
    header={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
    }
    url=f"https://15.push2.eastmoney.com/api/qt/clist/get?cb=jQuery112404243667054501892_1721739079856&pn={page}&pz=20&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&dect=1&wbp2u=|0|0|0|web&fid={cmd}&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152"
    response=requests.get(url,headers=header)
    content=response.content.decode("utf-8")
    if re.search('"data":null',content):
        dictlist=[]
    else:
        pat=re.compile('"diff":(\[.*?])')
        dictlist=re.findall(pat,content)[0] #返回20个词典
        dictlist=eval(dictlist) #将字符串转换为字典
    return dictlist

def get_data(cmd):
    for i in cmd.keys():
        page=0
        stocks=[]
        while True:
            page+=1
            dictlist=get_url(cmd[i],page)
            if dictlist!=[]:
                print("正在爬取"+i+"第"+str(page)+"页")
                df=dictlist
                for dicts in df:
                    dict={
                        "代码":dicts["f12"],
                        "名称":dicts["f14"],
                        "最新价":dicts["f2"],
                        "涨跌幅":dicts["f3"],
                        "涨跌额":dicts["f4"],
                        "成交量 （手）":dicts["f5"],
                        "成交额 (千元)":dicts["f6"],
                        "振幅 （%）":dicts["f7"],
                        "最高":dicts["f15"],
                        "最低":dicts["f16"],
                        "今开":dicts["f17"],
                        "昨收":dicts["f18"],
                        "量比":dicts["f10"],
                        "换手率":dicts["f8"],
                        "市盈率（动态）":dicts["f9"],
                        "市净率":dicts["f23"],
                    }
                    stocks.append(dict)
            else:
                break
        df=pd.DataFrame(stocks)
        df.to_excel(i+".xlsx",index=False)
def main():


    cmd={
        "沪深京A股": "f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048",
        "上证A股": "f3&fs=m:1+t:2,m:1+t:23",
        "深证A股": "f3&fs=m:0+t:6,m:0+t:80",
        "北证A股": "f3&fs=m:0+t:81+s:2048",
        "新股": "f26&fs=m:0+f:8,m:1+f:8",
        "创业板": "f3&fs=m:0+t:80",
        "科创板": "f3&fs=m:1+t:23",
        "沪股通": "f26&fs=b:BK0707",
        "深股通": "f26&fs=b:BK0804",
        "B股": "f3&fs=m:0+t:7,m:1+t:3",
        "风险警示板": "f3&fs=m:0+f:4,m:1+f:4",
    }
    get_data(cmd)

if __name__=="__main__":
    main()