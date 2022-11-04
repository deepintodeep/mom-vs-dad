import os
import zipfile
import shutil
from tqdm import tqdm

""" original """
# |___ 01.데이터
#      |___ 1.Training
#           |___ 라벨링데이터
#                |___ TL00001.zip
#                     |___ A(친가)
#                          |___ 1.Family
#                          |___ 2.Individual
#                          |___ 3.Age
#                     |___ B(외가)
#                          |___ 1.Family
#                          |___ 2.Individual
#                               |___ FOO01_IND_D_18_0_01.json
#                               |___ FOO01_IND_D_18_0_02.json
#                               ...
#                          |___ 3.Age
#           |___ 원천데이터
#                |___ TL00001.zip
#                     |___ A(친가)
#                          |___ 1.Family
#                          |___ 2.Individual
#                          |___ 3.Age
#                     |___ B(외가)
#                          |___ 1.Family
#                          |___ 2.Individual
#                               |___ FOO01_IND_D_18_0_01.jpg
#                               |___ FOO01_IND_D_18_0_02.jpg
#                               ...
#                          |___ 3.Age
#      |___ 2.Validation
#           ...


""" target """
# |___ data
#      |___ train
#           |___ images
#                |___ FOO01_IND_D_18_0_01.jpg
#                |___ FOO01_IND_D_18_0_02.jpg
#                ...
#           |___ labels
#                |___ FOO01_IND_D_18_0_01.json
#                |___ FOO01_IND_D_18_0_02.json
#                ...
#      |___ validation
#           ...



class moveFiles:
    # original zip file path
    _of = os.path.join(os.getcwd(), "01.데이터")
    orig_folder = {"train": {"images": os.path.join(_of, "1.Training/원천데이터"),
                             "labels": os.path.join(_of, "1.Training/라벨링데이터")},
                   "val":   {"images": os.path.join(_of, "2.Validation/원천데이터"),
                             "labels": os.path.join(_of, "2.Validation/라벨링데이터")}}

    # target zip file path
    _tf = os.path.join(os.getcwd(), "data")
    target_folder = {"train": {"images": os.path.join(_tf, "train/images"),
                               "labels": os.path.join(_tf, "train/labels")},
                     "val":   {"images": os.path.join(_tf, "validation/images"),
                               "labels": os.path.join(_tf, "validation/labels")}}

    def extract_zip(self, path):
        iddic = {'라벨링데이터' : "labels", '원천데이터' : "images"}
        idx = path.split('/')[-1]

        # temp folder file
        tmp_path = os.path.join(os.getcwd(), "data", "tmp", iddic[idx])

        # zip file list of the path folder
        ziplist = os.listdir(path)

        # extract only "2.individulal"
        for file in ziplist:
            filepath = os.path.join(path, file)

            # open zip files only
            try :
                zf = zipfile.ZipFile(filepath, 'r')
            except zipfile.BadZipFile:
                continue

            # encoding change
            # zipInfo = zf.infolist()
            # for member in zipInfo:
            #         member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')

            # filename list including '2.Individuals/'
            tmp_list = zf.infolist()
            individual_list = [i for i in tmp_list if '2.Individuals/' in i.filename][2:]

            # extract files
            for filename in individual_list:
                if filename:
                    zf.extract(filename, tmp_path)

    def move_folder_from_tmp(self, target_folder):
        # make target folder
        os.makedirs(target_folder, exist_ok=True)

        folders = ["A(─ú░í)", "B(┐▄░í)"]     # A(친가), A(외가)
        idx = target_folder.split('/')[-1]

        # tmp folder path including subdir
        tmp_path = os.path.join(os.getcwd(), "data", "tmp", idx)
        subdir = [os.path.join(tmp_path, folders[0], '2.Individuals'),
                  os.path.join(tmp_path, folders[1], '2.Individuals')]

        # files of tmp folder, [[],[]]
        files = [os.listdir(i) for i in subdir]

        # move files from tmp to target
        for i in range(2):
            for file in files[i]:
                nowpath = os.path.join(subdir[i], file)
                shutil.move(nowpath, target_folder)

    def reconst(self):
        idx1 = ["train", "val"]
        idx2 = ["images", "labels"]

        for i in range(2):
            for j in tqdm(range(2)):
                self.extract_zip(self.orig_folder[idx1[i]][idx2[j]])
                self.move_folder_from_tmp(self.target_folder[idx1[i]][idx2[j]])

        # remove tmp folder
        shutil.rmtree(os.path.join(os.getcwd(), "data", "tmp"))


if __name__ == '__main__' :
    a = moveFiles()
    a.reconst()