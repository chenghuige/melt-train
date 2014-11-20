#COMAKE2 edit-mode: -*- Makefile -*-
####################64Bit Mode####################
ifeq ($(shell uname -m),x86_64)
CC=/opt/compiler/gcc-4.8.2/bin/g++
CXX=/opt/compiler/gcc-4.8.2/bin/g++
CXXFLAGS=-g \
  -O5 \
  -pipe \
  -W \
  -Wall \
  -fPIC \
  -DHAVE_NETINET_IN_H \
  -Wno-unused-parameter \
  -Wno-deprecated \
  -std=c++11 \
  -fpermissive \
  -Wno-write-strings \
  -Wno-literal-suffix \
  -Wno-unused-local-typedefs \
  -fopenmp
CFLAGS=-g \
  -O5 \
  -pipe \
  -W \
  -Wall \
  -fPIC \
  -DHAVE_NETINET_IN_H \
  -Wno-unused-parameter \
  -Wno-deprecated \
  -std=c++11 \
  -fpermissive \
  -Wno-write-strings \
  -Wno-literal-suffix \
  -Wno-unused-local-typedefs \
  -fopenmp
CPPFLAGS=-D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\"
INCPATH=-I./ \
  -I./include \
  -I./utils \
  -I./include/Trainers/
DEP_INCPATH=-I../../../../app/search/sep/anti-spam/gezi \
  -I../../../../app/search/sep/anti-spam/gezi/include \
  -I../../../../app/search/sep/anti-spam/gezi/output \
  -I../../../../app/search/sep/anti-spam/gezi/output/include \
  -I../../../../app/search/sep/anti-spam/melt \
  -I../../../../app/search/sep/anti-spam/melt/include \
  -I../../../../app/search/sep/anti-spam/melt/output \
  -I../../../../app/search/sep/anti-spam/melt/output/include \
  -I../../../../com/btest/gtest \
  -I../../../../com/btest/gtest/include \
  -I../../../../com/btest/gtest/output \
  -I../../../../com/btest/gtest/output/include \
  -I../../../../lib2-64/bsl \
  -I../../../../lib2-64/bsl/include \
  -I../../../../lib2-64/bsl/output \
  -I../../../../lib2-64/bsl/output/include \
  -I../../../../lib2-64/ccode \
  -I../../../../lib2-64/ccode/include \
  -I../../../../lib2-64/ccode/output \
  -I../../../../lib2-64/ccode/output/include \
  -I../../../../lib2-64/dict \
  -I../../../../lib2-64/dict/include \
  -I../../../../lib2-64/dict/output \
  -I../../../../lib2-64/dict/output/include \
  -I../../../../lib2-64/libcrf \
  -I../../../../lib2-64/libcrf/include \
  -I../../../../lib2-64/libcrf/output \
  -I../../../../lib2-64/libcrf/output/include \
  -I../../../../lib2-64/others-ex \
  -I../../../../lib2-64/others-ex/include \
  -I../../../../lib2-64/others-ex/output \
  -I../../../../lib2-64/others-ex/output/include \
  -I../../../../lib2-64/postag \
  -I../../../../lib2-64/postag/include \
  -I../../../../lib2-64/postag/output \
  -I../../../../lib2-64/postag/output/include \
  -I../../../../lib2-64/ullib \
  -I../../../../lib2-64/ullib/include \
  -I../../../../lib2-64/ullib/output \
  -I../../../../lib2-64/ullib/output/include \
  -I../../../../lib2-64/wordseg \
  -I../../../../lib2-64/wordseg/include \
  -I../../../../lib2-64/wordseg/output \
  -I../../../../lib2-64/wordseg/output/include \
  -I../../../../public/comlog-plugin \
  -I../../../../public/comlog-plugin/include \
  -I../../../../public/comlog-plugin/output \
  -I../../../../public/comlog-plugin/output/include \
  -I../../../../public/configure \
  -I../../../../public/configure/include \
  -I../../../../public/configure/output \
  -I../../../../public/configure/output/include \
  -I../../../../public/connectpool \
  -I../../../../public/connectpool/include \
  -I../../../../public/connectpool/output \
  -I../../../../public/connectpool/output/include \
  -I../../../../public/odict \
  -I../../../../public/odict/include \
  -I../../../../public/odict/output \
  -I../../../../public/odict/output/include \
  -I../../../../public/spreg \
  -I../../../../public/spreg/include \
  -I../../../../public/spreg/output \
  -I../../../../public/spreg/output/include \
  -I../../../../public/uconv \
  -I../../../../public/uconv/include \
  -I../../../../public/uconv/output \
  -I../../../../public/uconv/output/include \
  -I../../../../quality/autotest/reportlib/cpp \
  -I../../../../quality/autotest/reportlib/cpp/include \
  -I../../../../quality/autotest/reportlib/cpp/output \
  -I../../../../quality/autotest/reportlib/cpp/output/include \
  -I../../../../third-64/boost \
  -I../../../../third-64/boost/include \
  -I../../../../third-64/boost/output \
  -I../../../../third-64/boost/output/include \
  -I../../../../third-64/gflags \
  -I../../../../third-64/gflags/include \
  -I../../../../third-64/gflags/output \
  -I../../../../third-64/gflags/output/include \
  -I../../../../third-64/glog \
  -I../../../../third-64/glog/include \
  -I../../../../third-64/glog/output \
  -I../../../../third-64/glog/output/include \
  -I../../../../third-64/gtest \
  -I../../../../third-64/gtest/include \
  -I../../../../third-64/gtest/output \
  -I../../../../third-64/gtest/output/include \
  -I../../../../third-64/libcurl \
  -I../../../../third-64/libcurl/include \
  -I../../../../third-64/libcurl/output \
  -I../../../../third-64/libcurl/output/include \
  -I../../../../third-64/pcre \
  -I../../../../third-64/pcre/include \
  -I../../../../third-64/pcre/output \
  -I../../../../third-64/pcre/output/include \
  -I../../../../third-64/tcmalloc \
  -I../../../../third-64/tcmalloc/include \
  -I../../../../third-64/tcmalloc/output \
  -I../../../../third-64/tcmalloc/output/include \
  -I../../../../third-64/zlib \
  -I../../../../third-64/zlib/include \
  -I../../../../third-64/zlib/output \
  -I../../../../third-64/zlib/output/include

#============ CCP vars ============
CCHECK=@ccheck.py
CCHECK_FLAGS=
PCLINT=@pclint
PCLINT_FLAGS=
CCP=@ccp.py
CCP_FLAGS=


#COMAKE UUID
COMAKE_MD5=3789faca693f0aa9c52584b1fa141c11  COMAKE


.PHONY:all
all:comake2_makefile_check libmelt_trainers.a 
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mall[0m']"
	@echo "make all done"

.PHONY:comake2_makefile_check
comake2_makefile_check:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mcomake2_makefile_check[0m']"
	#in case of error, update 'Makefile' by 'comake2'
	@echo "$(COMAKE_MD5)">comake2.md5
	@md5sum -c --status comake2.md5
	@rm -f comake2.md5

.PHONY:ccpclean
ccpclean:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mccpclean[0m']"
	@echo "make ccpclean done"

.PHONY:clean
clean:ccpclean
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mclean[0m']"
	rm -rf libmelt_trainers.a
	rm -rf ./output/lib/libmelt_trainers.a
	rm -rf src/Trainers/melt_trainers_FastRank.o
	rm -rf src/Trainers/melt_trainers_LinearSVM.o
	rm -rf src/Trainers/melt_trainers_SofiaTrainer.o
	rm -rf src/Trainers/melt_trainers_ThirdTrainer.o
	rm -rf src/Trainers/melt_trainers_Trainer.o
	rm -rf src/Trainers/melt_trainers_TrainerFactory.o
	rm -rf src/Trainers/melt_trainers_VWTrainer.o

.PHONY:dist
dist:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mdist[0m']"
	tar czvf output.tar.gz output
	@echo "make dist done"

.PHONY:distclean
distclean:clean
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mdistclean[0m']"
	rm -f output.tar.gz
	@echo "make distclean done"

.PHONY:love
love:
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mlove[0m']"
	@echo "make love done"

libmelt_trainers.a:src/Trainers/melt_trainers_FastRank.o \
  src/Trainers/melt_trainers_LinearSVM.o \
  src/Trainers/melt_trainers_SofiaTrainer.o \
  src/Trainers/melt_trainers_ThirdTrainer.o \
  src/Trainers/melt_trainers_Trainer.o \
  src/Trainers/melt_trainers_TrainerFactory.o \
  src/Trainers/melt_trainers_VWTrainer.o
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mlibmelt_trainers.a[0m']"
	ar crs libmelt_trainers.a src/Trainers/melt_trainers_FastRank.o \
  src/Trainers/melt_trainers_LinearSVM.o \
  src/Trainers/melt_trainers_SofiaTrainer.o \
  src/Trainers/melt_trainers_ThirdTrainer.o \
  src/Trainers/melt_trainers_Trainer.o \
  src/Trainers/melt_trainers_TrainerFactory.o \
  src/Trainers/melt_trainers_VWTrainer.o
	mkdir -p ./output/lib
	cp -f --link libmelt_trainers.a ./output/lib

src/Trainers/melt_trainers_FastRank.o:src/Trainers/FastRank.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_FastRank.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_FastRank.o src/Trainers/FastRank.cpp

src/Trainers/melt_trainers_LinearSVM.o:src/Trainers/LinearSVM.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_LinearSVM.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_LinearSVM.o src/Trainers/LinearSVM.cpp

src/Trainers/melt_trainers_SofiaTrainer.o:src/Trainers/SofiaTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_SofiaTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_SofiaTrainer.o src/Trainers/SofiaTrainer.cpp

src/Trainers/melt_trainers_ThirdTrainer.o:src/Trainers/ThirdTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_ThirdTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_ThirdTrainer.o src/Trainers/ThirdTrainer.cpp

src/Trainers/melt_trainers_Trainer.o:src/Trainers/Trainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_Trainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_Trainer.o src/Trainers/Trainer.cpp

src/Trainers/melt_trainers_TrainerFactory.o:src/Trainers/TrainerFactory.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_TrainerFactory.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_TrainerFactory.o src/Trainers/TrainerFactory.cpp

src/Trainers/melt_trainers_VWTrainer.o:src/Trainers/VWTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40msrc/Trainers/melt_trainers_VWTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -O3 \
  -DNDEBUG $(CXXFLAGS)  -o src/Trainers/melt_trainers_VWTrainer.o src/Trainers/VWTrainer.cpp

endif #ifeq ($(shell uname -m),x86_64)


