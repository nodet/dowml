#
# Download the latest checked build, and upload it on Pypi
#
set -e
ForReal=$1

if [ "$ForReal" != "real" ]; then
  TAG_NAME_SUFFIX="-test"
  PYPI_REPO="--repository testpypi"
else
  TAG_NAME_SUFFIX="-release"
fi

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
TAGNAME=V${VERSION}${TAG_NAME_SUFFIX}

if git rev-parse ${TAGNAME} > /dev/null 2>&1; then
  echo "Error: the tag '${TAGNAME}' already exists!"
  exit 1
fi

# Upload and tag
python3 -m twine upload ${PYPI_REPO} dist-checked/dowml-*
git tag ${TAGNAME} ${GITID}
git push public --tags
