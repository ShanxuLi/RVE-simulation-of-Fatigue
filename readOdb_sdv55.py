from odbAccess import openOdb
import odbAccess
import math
import numpy as np
import os

def readOdb(odbpath, odbname, key_FIP, key_EVOL, key_Part, storename):
    odbfilepath = odbpath+odbname+".odb"
    myOdb = odbAccess.openOdb(path=odbfilepath)
    myinstance = myOdb.rootAssembly.instances.values()[0]
    mystep = myOdb.steps.values()[-1]
    start_frame = len(mystep.frames)-1
    last_frame = len(mystep.frames)
    dataDir = odbpath+"/./"
    if not os.path.isdir(dataDir):
        os.system("mkdir -p "+dataDir)
    for i in range(start_frame, last_frame):
        if i < 10:
            counter = '00'+str(i)
        if i > 9 and i < 100:
            counter = '0'+str(i)
        else:
            counter = str(i)

        if not os.path.isfile(dataDir + 'data_FIP_{}.csv'.format(storename)):
            try:
                el_IDs_FIP = list()
                FIP = list()
                for value in mystep.frames[i].fieldOutputs[key_FIP].values:
                    el_IDs_FIP.append(value.elementLabel)
                    FIP.append(value.data)

                data_FIP = [el_IDs_FIP,FIP]
                data_FIP = list(zip(*data_FIP))

                with open(dataDir +'data_FIP_{}.csv'.format(storename), 'wb+') as f:
                    f.write('element_id,FIP\n')
                    for line in data_FIP:
                        f.write(str(line)[1:-1])
                        f.write('\n')
                print('data_FIP.csv has been successfully created')
            except:
                print(key_FIP +' not found')
        else:
            print('data_FIP.csv has already been successfully created')

        if not os.path.isfile(dataDir + 'data_el_vol_{}.csv'.format(storename)):
            try:
                el_IDs_vol = list()
                el_vol = list()
                for value in mystep.frames[i].fieldOutputs[key_EVOL].values:
                    el_IDs_vol.append(value.elementLabel)
                    el_vol.append(value.data)

                data_el_vol = [el_IDs_vol,el_vol]
                data_el_vol = list(zip(*data_el_vol))

                with open(dataDir + 'data_el_vol_{}.csv'.format(storename), 'wb+') as f:
                    f.write('element_id,el_vol\n')
                    for line in data_el_vol:
                        f.write(str(line)[1:-1])
                        f.write('\n')
                print('data_el_vol.csv has been successfully created')
            except:
                print(key_EVOL+' not found')
        else:
            print('data_el_vol.csv has already been successfully created')

    try:
        grainIDs = list()
        elementIDs = list()
        grainID = 0
        for sectionAssignment in myinstance.sectionAssignments:
            for element in sectionAssignment.region.elements:
                elementIDs.append(element.label)
                grainIDs.append(grainID)
            grainID +=1

        data_grainID = [elementIDs, grainIDs]
        data_grainID = list(zip(*data_grainID))
        print(dataDir + 'data_grain_ID_{}.csv')
        with open(dataDir + 'data_grain_ID_{}.csv'.format(storename), 'wb+') as f:
            f.write('element_id,grainID\n')
            for line in data_grainID:
                f.write(str(line)[1:-1])
                f.write('\n')
        print('data_grain_ID.csv has been successfully created')
    except:
        print(key_Part+' not found')


if __name__ == "__main__":
    odbpath = sys.argv[-3]
    odbname = sys.argv[-2]
    storename = sys.argv[-1]
    key_FIP = 'SDV55'
    key_EVOL = 'EVOL'
    key_Part = 'PART-1-1'
    readOdb(odbpath, odbname, key_FIP, key_EVOL, key_Part, storename)
