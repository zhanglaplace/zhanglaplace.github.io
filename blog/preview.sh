rm -rf .deploy_git
if [[ $? -eq 0 ]];then
   echo "Remove OK"
else
   echo "remove occur an Error"
fi
git config --global core.autocrlf false
hexo clean
hexo g
hexo s
if [[ $? -eq 0 ]];then
   echo "Upload OK"
else
   echo "Occur an Error"
fi
