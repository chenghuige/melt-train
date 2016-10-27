#COMAKE2 edit-mode: -*- Makefile -*-
####################64Bit Mode####################
ifeq ($(shell uname -m),x86_64)
CC=/opt/compiler/gcc-4.8.2/bin/g++
CXX=/opt/compiler/gcc-4.8.2/bin/g++
CXXFLAGS=-g \
  -O3 \
  -pipe \
  -W \
  -fPIC \
  -DHAVE_NETINET_IN_H \
  -Wno-unused-parameter \
  -Wno-deprecated \
  -std=c++11 \
  -fpermissive \
  -Wno-write-strings \
  -Wno-literal-suffix \
  -Wno-unused-local-typedefs \
  -Wreorder \
  -fopenmp
CFLAGS=-g \
  -O3 \
  -pipe \
  -W \
  -fPIC \
  -DHAVE_NETINET_IN_H \
  -Wno-unused-parameter \
  -Wno-deprecated \
  -std=c++11 \
  -fpermissive \
  -Wno-write-strings \
  -Wno-literal-suffix \
  -Wno-unused-local-typedefs \
  -Wreorder \
  -fopenmp
CPPFLAGS=-D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\"
INCPATH=-I../ \
  -I../include \
  -I../utils
DEP_INCPATH=-I../../../../../app/search/sep/anti-spam/gezi \
  -I../../../../../app/search/sep/anti-spam/gezi/include \
  -I../../../../../app/search/sep/anti-spam/gezi/output \
  -I../../../../../app/search/sep/anti-spam/gezi/output/include \
  -I../../../../../app/search/sep/anti-spam/gezi/third \
  -I../../../../../app/search/sep/anti-spam/gezi/third/include \
  -I../../../../../app/search/sep/anti-spam/gezi/third/output \
  -I../../../../../app/search/sep/anti-spam/gezi/third/output/include \
  -I../../../../../app/search/sep/anti-spam/gezi/third/rabit \
  -I../../../../../app/search/sep/anti-spam/gezi/third/rabit/include \
  -I../../../../../app/search/sep/anti-spam/gezi/third/rabit/output \
  -I../../../../../app/search/sep/anti-spam/gezi/third/rabit/output/include \
  -I../../../../../app/search/sep/anti-spam/melt \
  -I../../../../../app/search/sep/anti-spam/melt/include \
  -I../../../../../app/search/sep/anti-spam/melt/output \
  -I../../../../../app/search/sep/anti-spam/melt/output/include \
  -I../../../../../baidu/nlp-dnn/liblego \
  -I../../../../../baidu/nlp-dnn/liblego/include \
  -I../../../../../baidu/nlp-dnn/liblego/output \
  -I../../../../../baidu/nlp-dnn/liblego/output/include \
  -I../../../../../com/btest/gtest \
  -I../../../../../com/btest/gtest/include \
  -I../../../../../com/btest/gtest/output \
  -I../../../../../com/btest/gtest/output/include \
  -I../../../../../lib2-64/bsl \
  -I../../../../../lib2-64/bsl/include \
  -I../../../../../lib2-64/bsl/output \
  -I../../../../../lib2-64/bsl/output/include \
  -I../../../../../lib2-64/ccode \
  -I../../../../../lib2-64/ccode/include \
  -I../../../../../lib2-64/ccode/output \
  -I../../../../../lib2-64/ccode/output/include \
  -I../../../../../lib2-64/dict \
  -I../../../../../lib2-64/dict/include \
  -I../../../../../lib2-64/dict/output \
  -I../../../../../lib2-64/dict/output/include \
  -I../../../../../lib2-64/libcrf \
  -I../../../../../lib2-64/libcrf/include \
  -I../../../../../lib2-64/libcrf/output \
  -I../../../../../lib2-64/libcrf/output/include \
  -I../../../../../lib2-64/others-ex \
  -I../../../../../lib2-64/others-ex/include \
  -I../../../../../lib2-64/others-ex/output \
  -I../../../../../lib2-64/others-ex/output/include \
  -I../../../../../lib2-64/postag \
  -I../../../../../lib2-64/postag/include \
  -I../../../../../lib2-64/postag/output \
  -I../../../../../lib2-64/postag/output/include \
  -I../../../../../lib2-64/ullib \
  -I../../../../../lib2-64/ullib/include \
  -I../../../../../lib2-64/ullib/output \
  -I../../../../../lib2-64/ullib/output/include \
  -I../../../../../lib2-64/wordseg \
  -I../../../../../lib2-64/wordseg/include \
  -I../../../../../lib2-64/wordseg/output \
  -I../../../../../lib2-64/wordseg/output/include \
  -I../../../../../public/comlog-plugin \
  -I../../../../../public/comlog-plugin/include \
  -I../../../../../public/comlog-plugin/output \
  -I../../../../../public/comlog-plugin/output/include \
  -I../../../../../public/configure \
  -I../../../../../public/configure/include \
  -I../../../../../public/configure/output \
  -I../../../../../public/configure/output/include \
  -I../../../../../public/connectpool \
  -I../../../../../public/connectpool/include \
  -I../../../../../public/connectpool/output \
  -I../../../../../public/connectpool/output/include \
  -I../../../../../public/odict \
  -I../../../../../public/odict/include \
  -I../../../../../public/odict/output \
  -I../../../../../public/odict/output/include \
  -I../../../../../public/spreg \
  -I../../../../../public/spreg/include \
  -I../../../../../public/spreg/output \
  -I../../../../../public/spreg/output/include \
  -I../../../../../public/uconv \
  -I../../../../../public/uconv/include \
  -I../../../../../public/uconv/output \
  -I../../../../../public/uconv/output/include \
  -I../../../../../quality/autotest/reportlib/cpp \
  -I../../../../../quality/autotest/reportlib/cpp/include \
  -I../../../../../quality/autotest/reportlib/cpp/output \
  -I../../../../../quality/autotest/reportlib/cpp/output/include \
  -I../../../../../third-64/boost \
  -I../../../../../third-64/boost/include \
  -I../../../../../third-64/boost/output \
  -I../../../../../third-64/boost/output/include \
  -I../../../../../third-64/gflags \
  -I../../../../../third-64/gflags/include \
  -I../../../../../third-64/gflags/output \
  -I../../../../../third-64/gflags/output/include \
  -I../../../../../third-64/glog \
  -I../../../../../third-64/glog/include \
  -I../../../../../third-64/glog/output \
  -I../../../../../third-64/glog/output/include \
  -I../../../../../third-64/gtest \
  -I../../../../../third-64/gtest/include \
  -I../../../../../third-64/gtest/output \
  -I../../../../../third-64/gtest/output/include \
  -I../../../../../third-64/libcurl \
  -I../../../../../third-64/libcurl/include \
  -I../../../../../third-64/libcurl/output \
  -I../../../../../third-64/libcurl/output/include \
  -I../../../../../third-64/pcre \
  -I../../../../../third-64/pcre/include \
  -I../../../../../third-64/pcre/output \
  -I../../../../../third-64/pcre/output/include \
  -I../../../../../third-64/protobuf \
  -I../../../../../third-64/protobuf/include \
  -I../../../../../third-64/protobuf/output \
  -I../../../../../third-64/protobuf/output/include \
  -I../../../../../third-64/tcmalloc \
  -I../../../../../third-64/tcmalloc/include \
  -I../../../../../third-64/tcmalloc/output \
  -I../../../../../third-64/tcmalloc/output/include \
  -I../../../../../third-64/zlib \
  -I../../../../../third-64/zlib/include \
  -I../../../../../third-64/zlib/output \
  -I../../../../../third-64/zlib/output/include

#============ CCP vars ============
CCHECK=@ccheck.py
CCHECK_FLAGS=
PCLINT=@pclint
PCLINT_FLAGS=
CCP=@ccp.py
CCP_FLAGS=


#COMAKE UUID
COMAKE_MD5=b2c824eecd263c77d75331f1a1b49a63  COMAKE


.PHONY:all
all:comake2_makefile_check melt 
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
	rm -rf melt
	rm -rf ./output/bin/melt
	rm -rf melt_melt.o
	rm -rf ../src/Trainers/melt_EnsembleTrainer.o
	rm -rf ../src/Trainers/melt_Gbdt.o
	rm -rf ../src/Trainers/melt_LibLinearTrainer.o
	rm -rf ../src/Trainers/melt_LibSVMTrainer.o
	rm -rf ../src/Trainers/melt_LinearSVM.o
	rm -rf ../src/Trainers/melt_SofiaTrainer.o
	rm -rf ../src/Trainers/melt_ThirdTrainer.o
	rm -rf ../src/Trainers/melt_Trainer.o
	rm -rf ../src/Trainers/melt_TrainerFactory.o
	rm -rf ../src/Trainers/melt_VWTrainer.o
	rm -rf ../src/Run/melt_Melt.o
	rm -rf ../src/Prediction/Instances/melt_InstanceParser.o

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

melt:melt_melt.o \
  ../src/Trainers/melt_EnsembleTrainer.o \
  ../src/Trainers/melt_Gbdt.o \
  ../src/Trainers/melt_LibLinearTrainer.o \
  ../src/Trainers/melt_LibSVMTrainer.o \
  ../src/Trainers/melt_LinearSVM.o \
  ../src/Trainers/melt_SofiaTrainer.o \
  ../src/Trainers/melt_ThirdTrainer.o \
  ../src/Trainers/melt_Trainer.o \
  ../src/Trainers/melt_TrainerFactory.o \
  ../src/Trainers/melt_VWTrainer.o \
  ../src/Run/melt_Melt.o \
  ../src/Prediction/Instances/melt_InstanceParser.o
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mmelt[0m']"
	$(CXX) melt_melt.o \
  ../src/Trainers/melt_EnsembleTrainer.o \
  ../src/Trainers/melt_Gbdt.o \
  ../src/Trainers/melt_LibLinearTrainer.o \
  ../src/Trainers/melt_LibSVMTrainer.o \
  ../src/Trainers/melt_LinearSVM.o \
  ../src/Trainers/melt_SofiaTrainer.o \
  ../src/Trainers/melt_ThirdTrainer.o \
  ../src/Trainers/melt_Trainer.o \
  ../src/Trainers/melt_TrainerFactory.o \
  ../src/Trainers/melt_VWTrainer.o \
  ../src/Run/melt_Melt.o \
  ../src/Prediction/Instances/melt_InstanceParser.o -Xlinker "-("  ../../../../../app/search/sep/anti-spam/gezi/libgezi_common.a \
  ../../../../../app/search/sep/anti-spam/gezi/libgezi_cppformat.a \
  ../../../../../app/search/sep/anti-spam/gezi/libgezi_json.a \
  ../../../../../app/search/sep/anti-spam/gezi/output/lib/libPYNotation.a \
  ../../../../../app/search/sep/anti-spam/gezi/third/rabit/lib/librabit.a \
  ../../../../../app/search/sep/anti-spam/gezi/third/rabit/lib/librabit_base.a \
  ../../../../../app/search/sep/anti-spam/gezi/third/rabit/lib/librabit_mock.a \
  ../../../../../app/search/sep/anti-spam/melt/libblas.a \
  ../../../../../app/search/sep/anti-spam/melt/liblinear.a \
  ../../../../../app/search/sep/anti-spam/melt/libmelt.a \
  ../../../../../app/search/sep/anti-spam/melt/libsofia.a \
  ../../../../../app/search/sep/anti-spam/melt/libsvm.a \
  ../../../../../app/search/sep/anti-spam/melt/libvw.a \
  ../../../../../baidu/nlp-dnn/liblego/libliblego.a \
  ../../../../../com/btest/gtest/output/lib/libgtest.a \
  ../../../../../com/btest/gtest/output/lib/libgtest_main.a \
  ../../../../../lib2-64/bsl/lib/libbsl.a \
  ../../../../../lib2-64/bsl/lib/libbsl_ResourcePool.a \
  ../../../../../lib2-64/bsl/lib/libbsl_archive.a \
  ../../../../../lib2-64/bsl/lib/libbsl_buffer.a \
  ../../../../../lib2-64/bsl/lib/libbsl_check_cast.a \
  ../../../../../lib2-64/bsl/lib/libbsl_exception.a \
  ../../../../../lib2-64/bsl/lib/libbsl_pool.a \
  ../../../../../lib2-64/bsl/lib/libbsl_utils.a \
  ../../../../../lib2-64/bsl/lib/libbsl_var.a \
  ../../../../../lib2-64/bsl/lib/libbsl_var_implement.a \
  ../../../../../lib2-64/bsl/lib/libbsl_var_utils.a \
  ../../../../../lib2-64/ccode/lib/libulccode.a \
  ../../../../../lib2-64/dict/lib/libuldict.a \
  ../../../../../lib2-64/libcrf/lib/libcrf.a \
  ../../../../../lib2-64/others-ex/lib/libullib_ex.a \
  ../../../../../lib2-64/postag/lib/libpostag.a \
  ../../../../../lib2-64/ullib/lib/libullib.a \
  ../../../../../lib2-64/wordseg/libsegment.a \
  ../../../../../public/comlog-plugin/libcomlog.a \
  ../../../../../public/comlog-plugin/output/lib/libdfsappender.a \
  ../../../../../public/configure/libconfig.a \
  ../../../../../public/connectpool/libconnectpool.a \
  ../../../../../public/odict/libodict.a \
  ../../../../../public/spreg/libspreg.a \
  ../../../../../public/uconv/libuconv.a \
  ../../../../../quality/autotest/reportlib/cpp/libautotest.a \
  ../../../../../third-64/boost/lib/libboost_atomic.a \
  ../../../../../third-64/boost/lib/libboost_chrono.a \
  ../../../../../third-64/boost/lib/libboost_container.a \
  ../../../../../third-64/boost/lib/libboost_context.a \
  ../../../../../third-64/boost/lib/libboost_coroutine.a \
  ../../../../../third-64/boost/lib/libboost_date_time.a \
  ../../../../../third-64/boost/lib/libboost_exception.a \
  ../../../../../third-64/boost/lib/libboost_filesystem.a \
  ../../../../../third-64/boost/lib/libboost_graph.a \
  ../../../../../third-64/boost/lib/libboost_locale.a \
  ../../../../../third-64/boost/lib/libboost_math_c99.a \
  ../../../../../third-64/boost/lib/libboost_math_c99f.a \
  ../../../../../third-64/boost/lib/libboost_math_c99l.a \
  ../../../../../third-64/boost/lib/libboost_math_tr1.a \
  ../../../../../third-64/boost/lib/libboost_math_tr1f.a \
  ../../../../../third-64/boost/lib/libboost_math_tr1l.a \
  ../../../../../third-64/boost/lib/libboost_mpi.a \
  ../../../../../third-64/boost/lib/libboost_mpi_python.a \
  ../../../../../third-64/boost/lib/libboost_prg_exec_monitor.a \
  ../../../../../third-64/boost/lib/libboost_program_options.a \
  ../../../../../third-64/boost/lib/libboost_python.a \
  ../../../../../third-64/boost/lib/libboost_random.a \
  ../../../../../third-64/boost/lib/libboost_regex.a \
  ../../../../../third-64/boost/lib/libboost_serialization.a \
  ../../../../../third-64/boost/lib/libboost_signals.a \
  ../../../../../third-64/boost/lib/libboost_system.a \
  ../../../../../third-64/boost/lib/libboost_test_exec_monitor.a \
  ../../../../../third-64/boost/lib/libboost_thread.a \
  ../../../../../third-64/boost/lib/libboost_timer.a \
  ../../../../../third-64/boost/lib/libboost_unit_test_framework.a \
  ../../../../../third-64/boost/lib/libboost_wave.a \
  ../../../../../third-64/boost/lib/libboost_wserialization.a \
  ../../../../../third-64/gflags/lib/libgflags.a \
  ../../../../../third-64/gflags/lib/libgflags_nothreads.a \
  ../../../../../third-64/glog/lib/libglog.a \
  ../../../../../third-64/gtest/lib/libgtest.a \
  ../../../../../third-64/gtest/lib/libgtest_main.a \
  ../../../../../third-64/libcurl/lib/libcurl.a \
  ../../../../../third-64/pcre/lib/libpcre.a \
  ../../../../../third-64/pcre/lib/libpcrecpp.a \
  ../../../../../third-64/pcre/lib/libpcreposix.a \
  ../../../../../third-64/protobuf/lib/libprotobuf-lite.a \
  ../../../../../third-64/protobuf/lib/libprotobuf.a \
  ../../../../../third-64/protobuf/lib/libprotoc.a \
  ../../../../../third-64/tcmalloc/lib/libprofiler.a \
  ../../../../../third-64/tcmalloc/lib/libtcmalloc.a \
  ../../../../../third-64/tcmalloc/lib/libtcmalloc_and_profiler.a \
  ../../../../../third-64/tcmalloc/lib/libtcmalloc_debug.a \
  ../../../../../third-64/tcmalloc/lib/libtcmalloc_minimal.a \
  ../../../../../third-64/tcmalloc/lib/libtcmalloc_minimal_debug.a \
  ../../../../../third-64/zlib/lib/libz.a -lpthread \
  -g \
  -lgomp -Xlinker "-)" -o melt
	mkdir -p ./output/bin
	cp -f --link melt ./output/bin

melt_melt.o:melt.cc
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40mmelt_melt.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o melt_melt.o melt.cc

../src/Trainers/melt_EnsembleTrainer.o:../src/Trainers/EnsembleTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_EnsembleTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_EnsembleTrainer.o ../src/Trainers/EnsembleTrainer.cpp

../src/Trainers/melt_Gbdt.o:../src/Trainers/Gbdt.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_Gbdt.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_Gbdt.o ../src/Trainers/Gbdt.cpp

../src/Trainers/melt_LibLinearTrainer.o:../src/Trainers/LibLinearTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_LibLinearTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_LibLinearTrainer.o ../src/Trainers/LibLinearTrainer.cpp

../src/Trainers/melt_LibSVMTrainer.o:../src/Trainers/LibSVMTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_LibSVMTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_LibSVMTrainer.o ../src/Trainers/LibSVMTrainer.cpp

../src/Trainers/melt_LinearSVM.o:../src/Trainers/LinearSVM.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_LinearSVM.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_LinearSVM.o ../src/Trainers/LinearSVM.cpp

../src/Trainers/melt_SofiaTrainer.o:../src/Trainers/SofiaTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_SofiaTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_SofiaTrainer.o ../src/Trainers/SofiaTrainer.cpp

../src/Trainers/melt_ThirdTrainer.o:../src/Trainers/ThirdTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_ThirdTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_ThirdTrainer.o ../src/Trainers/ThirdTrainer.cpp

../src/Trainers/melt_Trainer.o:../src/Trainers/Trainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_Trainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_Trainer.o ../src/Trainers/Trainer.cpp

../src/Trainers/melt_TrainerFactory.o:../src/Trainers/TrainerFactory.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_TrainerFactory.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_TrainerFactory.o ../src/Trainers/TrainerFactory.cpp

../src/Trainers/melt_VWTrainer.o:../src/Trainers/VWTrainer.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Trainers/melt_VWTrainer.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Trainers/melt_VWTrainer.o ../src/Trainers/VWTrainer.cpp

../src/Run/melt_Melt.o:../src/Run/Melt.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Run/melt_Melt.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Run/melt_Melt.o ../src/Run/Melt.cpp

../src/Prediction/Instances/melt_InstanceParser.o:../src/Prediction/Instances/InstanceParser.cpp
	@echo "[[1;32;40mCOMAKE:BUILD[0m][Target:'[1;32;40m../src/Prediction/Instances/melt_InstanceParser.o[0m']"
	$(CXX) -c $(INCPATH) $(DEP_INCPATH) -D_GNU_SOURCE \
  -D__STDC_LIMIT_MACROS \
  -DVERSION=\"1.9.8.7\" \
  -DMELT_USE_THIRD_PARTY $(CXXFLAGS)  -o ../src/Prediction/Instances/melt_InstanceParser.o ../src/Prediction/Instances/InstanceParser.cpp

endif #ifeq ($(shell uname -m),x86_64)


