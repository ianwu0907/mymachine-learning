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
    error = True
    while (re.search('"data":null', content) and error==True):
        for i in range (1,10):
            url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get?cb=jQuery351043505281850346234_1721885857246&secid={i}.{stockn}&ut=fa5fd1943c7b386f172d6893dbfba10b&fields1=f1%2Cf2%2Cf3%2Cf4%2Cf5%2Cf6&fields2=f51%2Cf52%2Cf53%2Cf54%2Cf55%2Cf56%2Cf57%2Cf58%2Cf59%2Cf60%2Cf61&klt=101&fqt=1&end=20500101&lmt=1000000&_"
            response = requests.get(url, headers=header)
            content = response.content.decode("utf-8")
            if not re.search('"data":null', content):
                break
                error = False
        else:
            error = True
            stockn = str(input('你输入的股票有误，请重新输入股票代码：'))

    pat=re.compile('"klines":(\[.*?])')

    strlist=re.findall(pat,content)[0]

    strlist=eval(strlist)

    return strlist

def get_data(strlist,stockn):
    print("正在获取数据...")
    stockdata=[]
    dictlist=[]
    for i in strlist:
        stockdata.append(i.split(","))
    for stockdatas in stockdata:
        dict={
            "date":stockdatas[0],
            "open":float(stockdatas[1]),
            "close":float(stockdatas[2]),
            "high":float(stockdatas[3]),
            "low":float(stockdatas[4]),
            "vol":int(stockdatas[5]),
            "amount(thousand)":round(float(stockdatas[6])/1000,2),
            "amplitude(%)":float(stockdatas[7]),
            "pct_chg(%)":float(stockdatas[8]),
            "change":float(stockdatas[9]),
            "换手率(%)":float(stockdatas[10]),
        }
        # ignore all rows with 0 values
        # 检查是否有0值
        if all(value != 0 for value in dict.values()):
            dictlist.append(dict)  # 只有所有值都不为0时才添加

    df=pd.DataFrame(dictlist)

    df.to_excel(stockn+"_history"+".xlsx",index=False)
    df.to_csv(stockn+"_history"+".csv",index=False)


def main():
    stockn=str(input('请输入股票代码：'))
    strlist=get_url(stockn)
    get_data(strlist,stockn)

if __name__=="__main__":
    main()