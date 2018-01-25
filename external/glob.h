#ifndef TOOLS_BOOST_FILESYSTEM_FILTEREDDIRECTORYITERATOR_H_
#define TOOLS_BOOST_FILESYSTEM_FILTEREDDIRECTORYITERATOR_H_

#include <glob.h>
#include <vector>
#include <string>

namespace external {
    /*
    * POSIX Glob Functionality
    * source: https://gist.github.com/lucianmachado/9a26d5745497ffe5d054
    */
    inline std::vector<std::string> glob(const std::string& pat){
        using namespace std;

        glob_t glob_result;
        glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
        vector<string> ret;
        for(unsigned int i=0;i<glob_result.gl_pathc;++i){
            ret.push_back(string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        return ret;
    }
};

#endif