import os
import os.path as osp
import glob

"""
If file path is given, return it directly
For txt file, read it and return each line as file path
If it's a folder, return a list with names of each file

i_file_path:  1.indicative file:   /folder1/folder2/sample1.jpg
              2.file list:         /folder1/folder2/file_list.txt
              3.indicative folder: /folder1/folder2/
i_file_exts: [".jpg", ".png"]

reture: ["/folder1/folder2/sample1.jpg",
         "/folder1/folder2/sample2.jpg",
         ...
        ]
"""
def get_filelist2(i_file_path, i_file_exts=[".jpg"]):
    # input_path_extension = i_file_path.split('.')[-1]
    input_path_extension = os.path.splitext(i_file_path)[1]
    if input_path_extension in i_file_exts:
        return [i_file_path]
    elif input_path_extension == ".txt":
        with open(i_file_path, "r") as f:
            return f.read().splitlines()
    else:
        file_list = []
        for ext in i_file_exts:
            file_list = file_list + glob.glob(os.path.join(i_file_path, "*" + ext))
        return file_list

"""
If file path is given, return it directly
For txt file, read it and return each line as file path
If it's a folder, return a list with names of each file

input:
    i_file_path: the path include the images, it could be 3 type if format
        1. indicative image file : '/folder1/folder2/sample1.jpg'
        2. image list in txt file: '/folder1/folder2/file_list.txt'
        3. folder include images : '/folder1/folder2/'
    i_file_exts: filter file, only work in type 1、3
        ['.jpg', '.png']
    i_recursive: if walk throught all the folder, only work in type 3
        True or False(default)

reture: 
    file_list = 
        ["/folder1/folder2/sample1.jpg", "folder2", "sample1.jpg", "sample1"
         "/folder1/folder2/sample2.jpg", "folder2", "sample2.jpg", "sample2"
         ...
        ]
"""
def get_filelist(i_file_path, i_file_exts=[".jpg"], i_rtDir="", i_recursive=False):

    input_path_extension = os.path.splitext(i_file_path)[1]

    file_list_tmp = []
    if input_path_extension in i_file_exts:
        file_list_tmp = [i_file_path]
    elif input_path_extension == ".txt":
        with open(i_file_path, "r") as f:
            file_list_tmp = f.read().splitlines()
    else:
        if i_recursive == False:
            for ext in i_file_exts:
                # file_list_tmp = file_list_tmp + glob.glob(os.path.join(i_file_path, "*" + ext))
                file_list_tmp = file_list_tmp + glob.glob(f"{i_file_path}/*{ext}")
        else:
            walk_list = os.walk(i_file_path)
            for root, dirs, files in walk_list:
                for fl in files:
                    if osp.splitext(fl)[1] in i_file_exts:
                        file_list_tmp.append(f"{root}/{fl}")

    file_list = []
    for dirfile in file_list_tmp:
        dirfile = dirfile if i_rtDir == "" else f"{i_rtDir}/{dirfile}"
        ''' dirfile = "/media/Documents/01_Codebase/1000_Lo_codebase/2023_12_09.jpg" '''
        _dir = osp.dirname(dirfile) # e.g. "/media/Documents/01_Codebase/1000_Lo_codebase"
        subdir = _dir.split("/")[-1]
        file = osp.basename(dirfile) # e.g. "2023_12_09.jpg"
        filemain = file.split(".")[0]
        file_list.append([dirfile, _dir, subdir, file, filemain])

    return file_list