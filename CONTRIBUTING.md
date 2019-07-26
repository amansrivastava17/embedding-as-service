# How to submit an issue?

First, please see [contribution-guide.org](http://www.contribution-guide.org/) for the steps we expect from contributors before submitting an issue or bug report. Be as concrete as possible, include relevant logs, package versions etc.

**The proper place for open-ended questions is the [Email](amans.rlx@gmail.com).** Github is not the right place for research discussions or feature requests.

# How to add a new feature or create a pull request?

1. <a href="https://github.com/amansrivastava17/embedding-as-service/fork">Fork the Embedding-as-service repository</a>
2. Clone your fork: `git clone https://github.com/<YOUR_GITHUB_USERNAME>/embedding-as-service.git`
3. Create a new branch based on `develop`: `git checkout -b my-feature develop`
4. Setup your Python enviroment
   - Create a new [virtual environment](https://virtualenv.pypa.io/en/stable/): `pip install virtualenv; virtualenv embed_as_service` and activate it:
      - For linux: `source embed_as_service/bin/activate` 
      - For windows: `embed_as_service\Scripts\activate`
5. To add your own model or embeddings
    - create a new folder with the name as the model/feature name
    - create the `__init__.py` 
    - create a Embedding class and add all the details. you can use any other init file for reference.
    - create `_token` `load_model` and `encode` funtions. 
6. Check that everything's OK in your branch:
   - Check it for PEP8: `tox -e flake8`
   - Build its documentation (works only for MacOS/Linux): `tox -e docs` (documentation stored in `docs/src/_build`)
   - Run unit tests: `tox -e py{version}-{os}`, for example `tox -e py27-linux` or `tox -e py36-win` where
      - `{version}` is one of `27`, `35`, `36`
      - `{os}` is either `win` or `linux`
7. Add files, commit and push: `git add ... ; git commit -m "my commit message"; git push origin my-feature`
8. [Create a PR](https://help.github.com/articles/creating-a-pull-request/) on Github. Write a **clear description** for your PR, including all the context and relevant information, such as:
   - The issue that you fixed, e.g. `Fixes #123`
   - Motivation: why did you create this PR? What functionality did you set out to improve? What was the problem + an overview of how you fixed it? Whom does it affect and how should people use it?
   - Any other useful information: links to other related Github or mailing list issues and discussions, benchmark graphs, academic papers…

P.S. for developers: for details on the code style, CI, testing and similar we will be soon coming out with a blog.


**Thanks for Contributing!**
