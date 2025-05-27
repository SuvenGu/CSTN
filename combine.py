
import os
from osgeo import gdal, gdalconst
from osgeo import gdal
import numpy as np
import os 
from glob import glob
from math import ceil
import time
from osgeo import gdal,osr
from osgeo import ogr
np.set_printoptions(suppress=True)
def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x,max_y = geotrans[0],geotrans[3]
    max_x,min_y = geotrans[0]+xsize*geotrans[1],geotrans[3]+ysize*geotrans[5]
    ds = None
    return min_x,max_y,max_x,min_y

def RasterMosaic(file_list,outpath):
    Open = gdal.Open
    min_x,max_y,max_x,min_y=GetExtent(file_list[0])
    for infile in file_list:
        minx,maxy,maxx,miny = GetExtent(infile)
        min_x,min_y = min(min_x,minx),min(min_y,miny)
        max_x,max_y = max(max_x,maxx),max(max_y,maxy)
    
    in_ds = Open(file_list[0])
    in_band=in_ds.GetRasterBand(1)
    geotrans = list(in_ds.GetGeoTransform())
    width,height = geotrans[1],geotrans[5]
    columns = ceil((max_x-min_x)/width)#列数
    rows = ceil((max_y-min_y)/(-height))#行数
    
    outfile = outpath#结果文件名，可自行修改
    driver=gdal.GetDriverByName('GTiff')
    out_ds=driver.Create(outfile,columns,rows,1,in_band.DataType)
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0]=min_x#更正左上角坐标
    geotrans[3]=max_y
    out_ds.SetGeoTransform(geotrans)
    out_band=out_ds.GetRasterBand(1)
    inv_geotrans=gdal.InvGeoTransform(geotrans)

    for in_fn in file_list:
        in_ds=Open(in_fn)
        in_gt=in_ds.GetGeoTransform()
        offset=gdal.ApplyGeoTransform(inv_geotrans,in_gt[0],in_gt[3])
        x,y=map(int,offset)

        data=in_ds.GetRasterBand(1).ReadAsArray()
        out_band.WriteArray(data,x,y)#x，y是开始写入时左上角像元行列号
    del in_ds,out_band,out_ds
    return outfile

# 根据矢量裁剪图像
def extract_by_shp_feature(in_shp_path, feature_id, in_raster_path, out_raster_path):
    input_raster = gdal.Open(in_raster_path)

    # 利用gdal.Warp进行裁剪
    # https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Warp
    result = gdal.Warp(
        out_raster_path,
        input_raster,
        format='GTiff',
        cutlineDSName=in_shp_path,  # 用于裁剪的矢量
        cutlineWhere=f"NAME_1='{feature_id}'",  # 根据 feature_id 进行裁剪
        # cutlineWhere=f"COUNTRY='{feature_id}'",
        cropToCutline=True,  # 是否使用cutlineDSName的extent作为输出的界线
        dstNodata=255  # 输出数据的nodata值
    )
    result.FlushCache()
    del result

def projtif(intif,outtif,xres = 10,yres = 10):
    # 创建Albers等面积投影对象
    spatial_ref = osr.SpatialReference()

    ## 需要注意投影设置
    spatial_ref.ImportFromProj4("+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")  #中国
    # spatial_ref.ImportFromProj4("+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs") #美国
    
    gdal.Warp(outtif , # 输出栅格路径
            intif,  # 输入栅格
            format='GTiff',  # 导出tif格式
            dstSRS = spatial_ref, # 投影
            xRes=xres, # 重采样后的x方向分辨率
            yRes=yres, # 重采样后的y方向分辨率
            resampleAlg=gdal.GRA_NearestNeighbour, #最临近重采样
            creationOptions=['COMPRESS=LZW']#lzw压缩栅格
            )
    return outtif


def filter_vector_in_memory_and_cal(input_raster, in_shp_path,ids):
 # 创建Albers等面积投影对象
    spatial_ref = osr.SpatialReference()

    ## 需要注意投影设置
    # spatial_ref.ImportFromProj4("+proj=aea +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")  #中国
    # spatial_ref.ImportFromProj4("+proj=aea +lat_0=30 +lon_0=10 +lat_1=43 +lat_2=62 +x_0=0 +y_0=0 +ellps=intl +units=m +no_defs") #欧洲
    spatial_ref.ImportFromProj4("+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs") #美国
    print(30)
    AREA = 0
    for i in ids:
        result = gdal.Warp(
            "",
            input_raster,
            format='MEM',
            cutlineDSName=in_shp_path,  # 用于裁剪的矢量
            cutlineWhere=f"STASD_A='{i}'",  # 根据 feature_id 进行裁剪
            # cutlineWhere=f"NAME_2='{i}'",  # 根据 feature_id 进行裁剪
            # cutlineWhere=f"GID_1='{i}'",  # 根据 feature_id 进行裁剪
            # cutlineWhere=f"ENG_NAME='{i}'",  # 根据 feature_id 进行裁剪
            # cutlineWhere=f"HASC_1='{i}'",
            cropToCutline=True,  # 是否使用cutlineDSName的extent作为输出的界线
            dstNodata=255,  # 输出数据的nodata值
            # dstSRS="EPSG:5069",  # 投影 美国
            # dstSRS="ESRI:102017",  # 投影
            dstSRS=spatial_ref,
            resampleAlg=gdal.GRA_NearestNeighbour,
            xRes=30,
            yRes=30)  # 最临近重采样
        
        # 读取栅格数据为数组
        raster_array = result.ReadAsArray()
        value,count = np.unique(raster_array,return_counts=True)
        print(i)
        print(value)
        # print(count*10*10/10000000)
        # AREA= AREA+np.sum(count[:-1])*10*10/10000000
        # print(count*30*30/10000)
        # AREA= AREA+np.sum(count[:-1])*30*30/10000  #公顷
        print(count*30*30/10000000)
        AREA= AREA+np.sum(count[:-1])*30*30/10000000  #千公顷
    print(AREA)

        


        
year = "2019"
name = "ND"

# 输入TIFF文件夹的路径
input_folder =f"/mnt/d1/psw/maize/few_shot_cstn/ND_MF/"
# 输出拼接后的TIFF文件路径
output_file = f'/mnt/d1/psw/maize/few_shot_cstn/combine/{year}_{name}_PRED_CSTN-semi.tif'

# 获取输入文件夹中的所有TIFF文件
tiff_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.tif')]

# 输入文件列表
input_files = tiff_files
print("RasterMosaic!")
RasterMosaic(tiff_files,output_file)
in_shp_path = "/mnt/d1/psw/maize/shp/gadm41_USA_1.shp"
# in_shp_path = "/mnt/d16T/psw/crop_data/shp/gadm36_UKR_1.shp"
# in_shp_path = "/mnt/d1/psw/maize/shp/gadm36_CHN_1.shp"
# in_shp_path = "/mnt/d16T/psw/crop_data/shp/gadm41_CZE_0.shp"
# in_shp_path = "/mnt/d1/psw/maize/shp/gadm41_HUN_1.shp"
# in_shp_path = "/mnt/d16T/psw/crop_data/shp/2019年地级.shp"
out_raster_path =output_file
extract_by_shp_feature(in_shp_path, 'North Dakota',output_file, out_raster_path)
# extract_by_shp_feature(in_shp_path, 'Czechia',output_file, out_raster_path)

clip_shp = "/mnt/d1/psw/maize/shp/ASD_2012_20m.shp"
# clip_shp = "/mnt/d16T/psw/crop_data/shp/2021年县级.shp"
# clip_shp = "/mnt/d1/psw/maize/shp/gadm41_HUN_1.shp"
# clip_shp = "/mnt/d1/psw/maize/shp/gadm36_CHN_2.shp"

# ids =['Kharkiv']

# clip_shp="/mnt/d16T/psw/crop_data/shp/gadm41_CZE_1.shp"
ids = ["3810","3820","3830","3840","3850","3860","3870","3880","3890"] #nd
# # ids = ["0510","0520","0530","0540","0550","0560","0570","0580","0590"]
# ids = ["4510","4520","4530","4540","4550","4580"] #SC
# ids = ["2210","2220","2230","2240","2250","2260","2270","2280","2290"] #LA
# ids = ["Changchun","Jilin","Siping","Liaoyuan","Tonghua","Baishan","Songyuan","Baicheng","Yanbianchaoxianzu"]
# ids = ["Shenyang","Dalian","Anshan","Fushun","Benxi","Dandong","Jinzhou","Fuxin","Liaoyang","Panjin","Tieling","Chaoyang","Huludao"]
# ids = ["Zhengzhou","Kaifeng","Luoyang","Pingdingshan","Anyang","Hebi","Xinxiang","Jiaozuo","Puyang","Xuchang","Luohe","Sanmenxia","Nanyang","Shangqiu","Xinyang","Zhoukou","Zhumadian","Jiyuan shi"]
# ids = ["CZE.11_1","CZE.12_1","CZE.1_1","CZE.10_1","CZE.3_1","CZE.13_1","CZE.6_1","CZE.5_1","CZE.9_1","CZE.4_1","CZE.‘2_1","CZE.8_1","CZE.14_1","CZE.7_1"]
# ids = ["Jiaozuo"]
# ENG_NAME_ids = ["Xiuwu","Boai","Wuzhi","Wenxian","Qinyang","Mengzhou"]
# HASC_1 = ["HU.FE","HU.KE","HU.VE","HU.GS","HU.VA","HU.ZA","HU.BA","HU.SO","HU.TO","HU.BZ","HU.HE","HU.NO","HU.HB","HU.JN","HU.SZ","HU.BK","HU.BE","HU.CS"]
# ENG_NAME_ids = ["Zhongmu","Gongyi","Xingyang","Xinmi","Xinzheng","Dengfeng"]
# ENG_NAME_ids = ["Fugou","Xihua","Shangshui","Shenqiu","Dancheng","Taikang","Luyi"]
# # ids=["North Dakota"]
filter_vector_in_memory_and_cal(out_raster_path,clip_shp,ids)
print("done!")