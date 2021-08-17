#
# Download the latest checked build, and upload it on Test Pypi
#
set -e

if [ -d "dist-checked" ]; then
  echo "Error: remove 'dist-checked' first!"
  exit 1
fi

# Download the latest 'dist-checked' artifact, and put it in
# directory dist-checked
gh run download -n dist-checked -D dist-checked

# Get version information
VERSION=`ls -1 dist-checked/dowml-*.tar.gz | sed 's#dist-checked/dowml-##' | sed 's/.tar.gz//'`
GITID=`cat dist-checked/version.info`
echo "Found version" ${VERSION} "with git id" ${GITID}

# Tag and upload
git tag -f V${VERSION}-test ${GITID}
python3 -m twine upload --repository testpypi dist/*
git push public --tags