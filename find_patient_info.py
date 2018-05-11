# -*- coding: utf-8 -*-
# @Time    : 2018/5/11 19:32
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : find_patient_info.py
# @Software: PyCharm

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import fnmatch
import re

#查找文件夹中所有文件，列出，并且匹配*.xml##
count=0
fw=open("patients_have_drugs_info.txt",'w')
fcontent="/data1/data/TCGA/2017_3_16_clincial"
fcontent=fcontent+os.sep #相当于在后面加上一个/
for i in os.listdir(fcontent):
    dirname=fcontent+i  #dirname是不同癌症比方说ACC，BRCA等的文件夹
    matchname_prefix=i.lower() # for future usage
    dirname=dirname+os.sep
    for filename in os.listdir(dirname):
        if fnmatch.fnmatch(filename,"*.xml"):
            finalname=dirname+filename  #finalname就是已经确定了某一个癌种，并且找到了详细到每一个病人的信息
            # print(finalname)
            DOMTree=xml.dom.minidom.parse(finalname)
            collection=DOMTree.documentElement
            matchname=matchname_prefix+":patient"  #找到某种病人的根
            ######到这里为止都是希望遍历每一个癌症病人的xml文件，因此以上是找出病人文件
            patients=collection.getElementsByTagName(matchname)[0] #不同的癌症的标签都不一样，用了癌症:patient来标识,且每个样本只有一个patient，因此不用担心patient的index不是0
            drugname_root=patients.getElementsByTagName("rx:drugs")[0].childNodes  #每一个病人的drugs应用情况，注意drugs和drug的区别。下面是有drugs数据的才使用，没有drug或者drugs数据的则不使用
            if len(drugname_root)>0:
                # print(type(drugname_root))
                # print(finalname,file=fw)
                drugnamelist=patients.getElementsByTagName("rx:drugs")[0]  #getElementByTagName返回的是一个只有一个元素的Element列表，因此需要索引[0]
                drugnames=drugnamelist.getElementsByTagName("rx:drug")  #找到上一个节点的元素（Element），然后抽提这个元素的信息即可。一般病人都是联合用药，因此rx:drug会出现多个，需要遍历
                for manytags in drugnames:
                    if len(manytags.getElementsByTagName("rx:drug_name")[0].childNodes)>0:
                        drugname=manytags.getElementsByTagName("rx:drug_name")[0].childNodes[0].data
                        #######请注意：这里的列表可以适当地改变，也为此特地标记出来###
                        for drugdictname in [(133, 'Doxorubicin,Doxil,Rubex'), (134, 'Etoposide'), (150, 'Bicalutamide,ICI-176334'), (152, 'CP466722,[2-(6,7-dimethoxyquinazolin-4-yl)-5-(pyridin-2-yl)-2H-1,2,4-triazol-3-amine]'), (156, 'AZD6482,KIN001-193'), (159, 'HG-6-64-1,KIN001-206'), (165, 'DMOG,Dimethyloxalylglcine'), (175, 'PAC-1,PAC-1'), (176, 'IPA-3,IPA-3'), (179, '5-Fluorouracil,5-FU'), (182, 'Obatoclax,Mesylate'), (203, 'BMS345541,BMS345541'), (205, 'BMS-708163,Avagacestat'), (206, 'Ruxolitinib,INCB-18424'), (208, 'Ispinesib,Mesylate,SB-715992'), (211, 'TL-2-105'), (219, 'AT-7519'), (221, 'TAK-715,KIN001-201'), (222, 'BX-912,KIN001-175'), (223, 'ZSTK474,KIN001-167'), (224, 'AS605240,KIN001-173'), (225, 'Genentech,Cpd,10'), (226, 'GSK1070916'), (228, 'KIN001-102'), (230, 'GSK429286A'), (231, 'FMK,KIN001-242'), (235, 'QL-XII-47'), (238, 'CAL-101'), (252, 'WZ3105'), (253, 'XMD14-99'), (254, 'AC220,Quizartinib'), (255, 'CP724714'), (256, 'JW-7-24-1'), (257, 'NPK76-II-72-1'), (258, 'STF-62247'), (260, 'NG-25'), (261, 'TL-1-85'), (263, 'FR-180204'), (265, 'Tubastatin,A'), (266, 'Zibotentan,Zibotentan'), (271, 'VNLG/124,4-(Butanoyloxymethyl)phenyl-(2E,4E,6E,8E)-3,7-dimethyl-9-(2,6,6-trimethylcyclohex-1-enyl)nona-2,4,6,8-tetraenoate'), (272, 'AR-42,HDAC-42'), (275, 'I-BET-762,GSK525762A'), (276, 'CAY10603'), (279, 'BIX02189'), (281, 'CH5424802'), (286, 'KIN001-236'), (290, 'KIN001-260,Bayer,IKKb,inhibitor'), (292, 'Masitinib,AB1010'), (294, 'MPS-1-IN-1'), (295, 'BHG712,NVP-BHG712'), (298, 'OSI-930'), (299, 'OSI-027,A-1065'), (300, 'CX-5461'), (301, 'PHA-793887'), (302, 'PI-103'), (303, 'PIK-93'), (304, 'SB52334'), (305, 'TPCA-1'), (306, 'TG101348'), (309, 'Y-39983'), (310, 'YM201636'), (312, 'Tivozanib,AV-951'), (326, 'GSK690693'), (328, 'SNX-2112'), (329, 'QL-XI-92'), (330, 'XMD13-2'), (331, 'QL-X-138'), (332, 'XMD15-27'), (333, 'T0901317'), (341, 'EX-527'), (344, 'THZ-2-49'), (345, 'KIN001-270'), (346, 'THZ-2-102-1'), (1003, 'Camptothecin,7-Ethyl-10-Hydroxy-Camptothecin,SN-38'), (1004, 'Vinblastine,Vinblastine,sulphate'), (1007, 'Docetaxel,RP-56976'), (1008, 'Methotrexate'), (1010, 'Gefitinib,ZD-1839'), (1011, 'Navitoclax,ABT-263'), (1012, 'Vorinostat,SAHA'), (1014, 'RDEA119,RDEA119'), (1016, 'Temsirolimus,CCI-779'), (1020, 'Lenalidomide'), (1026, '17-AAG,tanespimycin'), (1028, 'VX-702'), (1030, 'KU-55933'), (1032, 'Afatinib,BIBW2992,,Tovok'), (1033, 'GDC0449,GDC0449'), (1036, 'PLX4720'), (1037, 'BX-795,BX,795'), (1038, 'NU-7441,KU-57788'), (1047, 'Nutlin-3a,(-),Nutlin-3a,(-)'), (1054, 'PD-0332991,Palbociclib,Isethionate'), (1057, 'BEZ235,NVP-BEZ235'), (1060, 'PD-0325901,PD-0325901'), (1061, 'SB590885'), (1149, 'TW,37'), (1170, 'CCT018159'), (1175, 'AG-014699,PF-01367338'), (1199, 'Tamoxifen'), (1241, 'CHIR-99021,CT,99021'), (1242, '(5Z)-7-Oxozeaenol'), (1259, 'Talazoparib,BMN-673'), (1261, 'rTRAIL'), (1262, 'UNC1215'), (1264, 'SGC0946'), (1371, 'PLX4720'), (1372, 'Trametinib,GSK1120212'), (1373, 'Dabrafenib,GSK2118436'), (1375, 'Temozolomide,Temodar'), (1377, 'Afatinib'), (1378, 'Bleomycin,(50,uM)'), (1494, 'SN-38,7-ethyl-10-hydroxy-camptothecin'), (1495, 'Olaparib,KU0059436,,AZD2281'), (1498, 'selumetinib'), (1526, 'RDEA119')]: # 现在挑选的是>0.618黄金分割点预测性能的药物，117个。
                            match=re.match(drugname,drugdictname[1],re.I|re.M)
                            if match:
                                fw_drugid=open(str(drugdictname[0])+'.txt','a')
                                # print('match')
                                count+=1
                                print(drugdictname[0],filename,file=fw)
                                print(finalname,file=fw_drugid)
                            ######################
                        #接下来是匹配获得的高性能模型的药物名字，并且储存文件名在txt中，最好用 药物名字_药物ID 命名
                        # print(drugname)
                # print("done")
print('number of final matched file is',count)

filenamedir='/data1/data/zengyanru/findout_followups'
filenamedir+=os.sep
for i in os.listdir(filenamedir):
    print(i)
    filename=filenamedir+i
    f_followup=open(filename,'r')
    filenamewrite=i.split(".")[0]+"_followup.txt"
    f_write_followup=open(filenamewrite,'w')
    f_write_followup.write("patient_sample_name\tvital_status\tdays_to_last_followup\tdays_to_last_known_alive\tdays_to_death\n")
    for xmlname in f_followup:
        xmlname=xmlname.strip("\n")
        DOMTree=xml.dom.minidom.parse(xmlname)
        collection=DOMTree.documentElement
        cancername = collection.nodeName.split(":")[0]
        cancer_patientname = cancername + ":patient"
        patients=collection.getElementsByTagName(cancer_patientname)[0]

        ##第一个数###
        vital_status=patients.getElementsByTagName("clin_shared:vital_status")
        if len(vital_status) > 0:
            vital_status = patients.getElementsByTagName("clin_shared:vital_status")[0].childNodes
            if len(vital_status) > 0:
                num1 = vital_status = patients.getElementsByTagName("clin_shared:vital_status")[0].childNodes[0].data
            else:
                num1 = "none"
        else:
            num1="none"
        ###第二个数###
        days_to_last_followup = patients.getElementsByTagName("clin_shared:days_to_last_followup")
        if len(days_to_last_followup) > 0:
            days_to_last_followup = patients.getElementsByTagName("clin_shared:days_to_last_followup")[0].childNodes  # 对应于num2
            if len(days_to_last_followup) > 0:
                num2 = patients.getElementsByTagName("clin_shared:days_to_last_followup")[0].childNodes[0].data
            else:
                num2 = "none"
        else:
            num2="none"
        ###第三个数###
        days_to_last_known_alive = patients.getElementsByTagName("clin_shared:days_to_last_known_alive")
        if len(days_to_last_known_alive) > 0:
            days_to_last_known_alive = patients.getElementsByTagName("clin_shared:days_to_last_known_alive")[0].childNodes
            if len(days_to_last_known_alive) > 0:
                num3 = patients.getElementsByTagName("clin_shared:days_to_last_known_alive")[0].childNodes[0].data
            else:
                num3 = "none"
        else:
            num3="none"
        ###第四个数###
        days_to_death = patients.getElementsByTagName("clin_shared:days_to_death")
        if len(days_to_death) > 0:
            days_to_death = patients.getElementsByTagName("clin_shared:days_to_death")[0].childNodes
            if len(days_to_death) > 0:
                num4 = vital_status = patients.getElementsByTagName("clin_shared:days_to_death")[0].childNodes[0].data
            else:
                num4 = "none"
        else:
            num4="none"

        print("%s\t%s\t%s\t%s\t%s" % (xmlname,num1, num2, num3, num4),file=f_write_followup)
        # print("%s\t%s\t%s\t%s\t%s" % (xmlname,num1, num2, num3, num4))

