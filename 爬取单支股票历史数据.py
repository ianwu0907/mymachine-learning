import re
import requests
import pandas as pd

def get_url(stockn):
    header={
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
    }

    url=f"https://push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery351043505281850346234_1721885857246&secid=0.{stockn}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt=1000000&_"
    response=requests.get(url,headers=header)
    content=response.content.decode("utf-8")


    pat=re.compile('"klines":(\[.*?])')
    strlist=re.findall(pat,content)[0]
    strlist=eval(strlist)

    return strlist

def get_data(strlist,stockn):
    stockdata=[]
    dictlist=[]
    for i in strlist:
        stockdata.append(i.split(","))
    for stockdatas in stockdata:
        dict={
            "日期":stockdatas[0],
            "开盘价":stockdatas[1],
            "收盘价":stockdatas[2],
            "最高价":stockdatas[3],
            "最低价":stockdatas[4],
            "成交量":stockdatas[5],
            "成交额":stockdatas[6],
            "振幅(%)":stockdatas[7],
            "涨跌幅(%)":stockdatas[8],
            "涨跌额":stockdatas[9],
            "换手率(%)":stockdatas[10],
        }
        dictlist.append(dict)
    df=pd.DataFrame(dictlist)
    df.to_excel(stockn+"_"+stockdata[0][0]+".xlsx",index=False)


def main():
    stockn=str(input('请输入股票代码：'))
    strlist=get_url(stockn)
    get_data(strlist,stockn)

if __name__=="__main__":
    main()