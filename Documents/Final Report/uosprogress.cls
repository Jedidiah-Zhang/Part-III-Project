%%
%% This is file `uosprogress.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% uosdocs.dtx  (with options: `progress')
%% 
%% Copyright (C) 2001 by Steve R. Gunn
%% Modifications Copyright (C) 2019 by Edward Longman
%% 
%% This file is part of the UoSDocs class distribution
%% 
\NeedsTeXFormat{LaTeX2e}[2007/02/26]
\ProvidesClass{uosprogress}
              [2022/08/15 v1.6
              LaTeX document class]
%% ------------ Class/Formating Adjustment ----------------------
%% Adjust the book class to match the requirements
%% Set spacing, line and paragraph options
%% Set LaTeX builder options (work break penalties etc.)
\def\baseclass{book}
\RequirePackage{xcolor}
\colorlet{linkBlue}{blue}
\colorlet{custGray}{gray}
\colorlet{chapRed}{red}
\DeclareOption{sotoncolour}{
\definecolor{sotonMarineBlue}{RGB}{1,67,89} % Soton marine blue (P 7469C)
  \definecolor{sotonGrey}{RGB}{153,153,166} % Soton grey (P 443C)
  \definecolor{sotonRed}{RGB}{171,18,16} % Soton Red (P 484C)
  \colorlet{linkBlue}{sotonMarineBlue}
  \colorlet{custGray}{sotonGrey}
  \colorlet{chapRed}{sotonRed}
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}
\def\@checkoptions#1#2{
  \edef\@curroptions{\@ptionlist{\@currname.\@currext}}
  \@tempswafalse
  \@tfor\@this:=#2\do{
    \@expandtwoargs\in@{,\@this,}{,\@curroptions,}
    \ifin@ \@tempswatrue \@break@tfor \fi}
  \let\@this\@empty
  \if@tempswa \else \PassOptionsToClass{#1}{\baseclass}\fi
}
\@checkoptions{11pt}{{10pt}{11pt}{12pt}}
\PassOptionsToClass{a4paper}{\baseclass}
\ProcessOptions\relax
\LoadClass{\baseclass}
\newcommand\bhrule{\typeout{------------------------------------------------------------------------------}}
\newcommand\btypeout[1]{\bhrule\typeout{\space #1}\bhrule}
%% \usepackage[T1]{fontspec}
\usepackage[defaultsans]{droidsans}
\usepackage{mathpazo}
%%\usepackage[T1]{fontenc} %This may not be necessary for english only text
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{2.0ex plus0.5ex minus0.2ex}
\usepackage{geometry}
\geometry{a4paper,
            % left=1.25in,
            % right=1.25in,
            hmarginratio=1:2,
            textwidth=146.5mm,
            top=0.6in,
            bottom=0.8in,
            headheight=20pt,
            headsep=0.25in,
            foot=9pt,
            footskip=0.3in,
            bindingoffset=0.5in,
            includeheadfoot}
\raggedbottom
\setlength{\topskip}{1\topskip \@plus 5\p@}
\doublehyphendemerits=10000       % No consecutive line hyphens.
\brokenpenalty=10000              % No broken words across columns/pages.
\widowpenalty=9999                % Almost no widows at bottom of page.
\clubpenalty=9999                 % Almost no orphans at top of page.
\interfootnotelinepenalty=9999    % Almost never break footnotes.
\usepackage{fancyhdr}
\fancyhead[LE]{\textrm\thepage}
\fancyhead[LO]{\fancyplain{}{\textsl{\rightmark}}}
\fancyhead[RE]{\fancyplain{}{\textsl{\leftmark}}}
\fancyhead[RO]{\textrm\thepage}
\chead{}\lfoot{}\rfoot{}\cfoot{}
\pagestyle{fancy}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyhead[OR]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
}
\usepackage{titlesec}
\titleformat{\chapter}[hang] 
{\normalfont\huge\bfseries}{\chaptertitlename\ \thechapter:}{1em}{} 
\renewcommand{\chaptermark}[1]{\btypeout{\thechapter.\space #1}\markboth{\chaptername\ \thechapter.\hspace{1em}#1}{}}
\renewcommand{\sectionmark}[1]{\markright{\thesection.\hspace{1em}#1}}
\renewcommand{\subsectionmark}[1]{}
\def\cleardoublepage{\clearpage\if@twoside \ifodd\c@page\else
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}
\def\cleartoeven{\clearpage\if@twoside \ifodd\c@page
\hbox{}
\thispagestyle{empty}
\newpage
\if@twocolumn\hbox{}\newpage\fi\fi\fi}
%% -------------------- Figure/Table/Eq/Listing Stying --------------------
%% Set the styling for non text elements of the document
\usepackage{amsmath,amsfonts,amssymb,amscd,amsthm,xspace}
\theoremstyle{plain}
\newtheorem{example}{Example}[chapter]
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{axiom}[theorem]{Axiom}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\usepackage[justification=centerlast,font=small,labelfont=sc]{caption}
\setlength{\captionmargin}{20pt}
\newcommand{\fref}[1]{Figure~\ref{#1}}
\newcommand{\tref}[1]{Table~\ref{#1}}
\newcommand{\eref}[1]{Equation~\ref{#1}}
\newcommand{\cref}[1]{Chapter~\ref{#1}}
\newcommand{\sref}[1]{Section~\ref{#1}}
\newcommand{\aref}[1]{Appendix~\ref{#1}}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\dbltopfraction}{.85}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{.75}
\setcounter{topnumber}{9}
\setcounter{bottomnumber}{9}
\setcounter{totalnumber}{20}
\setcounter{dbltopnumber}{9}
\usepackage{graphicx}
\usepackage{epstopdf}
\usepackage[]{subcaption}
\usepackage{booktabs}
\usepackage{rotating}
\usepackage{listings}
\lstset{captionpos=b,
        frame=tb,
        basicstyle=\scriptsize\ttfamily,
        showstringspaces=false,
        keepspaces=true}
\lstdefinestyle{matlab} {
        language=Matlab,
        keywordstyle=\color{blue},
        commentstyle=\color[rgb]{0.13,0.55,0.13}\em,
        stringstyle=\color[rgb]{0.7,0,0} }
\usepackage[pdfpagemode={UseOutlines},bookmarks=true,bookmarksopen=true,
  bookmarksopenlevel=0,bookmarksnumbered=true,plainpages=false,pdfpagelabels,
  colorlinks,linkcolor={linkBlue},citecolor={linkBlue},urlcolor={linkBlue},
  pdfstartview={FitV},unicode,breaklinks=true]{hyperref}
\pdfstringdefDisableCommands{
  \let\\\space
}
%% --------------------- Organisational Structure ----------------------
\newcommand*{\supervisor}[2][]{\def\supname{#2}\def\supplural{#1}}
\newcommand*{\examiner}[2][]{\def\examname{#2}\def\examplural{#1}}
\newcommand*{\degree}[1]{\def\degreename{#1}}
\newcommand*{\authors}[1]{\def\authornames{#1}}
\newcommand*{\qualifications}[1]{\def\quals{#1}}
\newcommand*{\addresses}[1]{\def\addressnames{#1}}
\newcommand*{\documentDoi}[1]{\def\doicode{#1}}
\newcommand*{\volume}[2]{\def\volno{#1}\def\volof{#2}}
\newcommand*{\orcidid}[1]{\def\orcid{#1}}
\newcommand*{\university}[1]{\def\univname{#1}}
\newcommand*{\UNIVERSITY}[1]{\def\UNIVNAME{#1}}
\newcommand*{\department}[1]{\def\deptname{#1}}
\newcommand*{\DEPARTMENT}[1]{\def\DEPTNAME{#1}}
\newcommand*{\group}[1]{\def\groupname{#1}}
\newcommand*{\GROUP}[1]{\def\GROUPNAME{#1}}
\newcommand*{\faculty}[1]{\def\facname{#1}}
\newcommand*{\FACULTY}[1]{\def\FACNAME{#1}}
\newcommand*{\subject}[1]{\def\subjectname{#1}}
\newcommand*{\keywords}[1]{\def\keywordnames{#1}}
%% --------------------- Organisational Structure ----------------------
\documentDoi{}
\supervisor  {}
\examiner    {}
\degree      {}
\authors     {}
\qualifications{}
\orcidid{}
\volume{}{}
\university  {\texorpdfstring{\href{http://www.southampton.ac.uk}
                {University of Southampton}}
                {University of Southampton}}
\UNIVERSITY  {\MakeUppercase{\univname}}
\department  {School of Electronics and Computer Science}
\DEPARTMENT  {\MakeUppercase{\deptname}}
\group       {}
\GROUP       {\MakeUppercase{\groupname}}
\faculty     {Faculty of Physical Sciences and Engineering}
\FACULTY     {\MakeUppercase{\facname}}
\addresses   {}
\subject     {}
\keywords    {}
\usepackage{titling}
\AtBeginDocument{
  \hypersetup{pdftitle={\thetitle}}
  \hypersetup{pdfsubject={\subjectname}}
  \hypersetup{pdfauthor={\authornames}}
  \hypersetup{pdfkeywords={\keywordnames}}
}
\renewcommand\maketitle{
  \btypeout{Title Page}
  \thispagestyle{empty}
  \begin{titlepage}
    \let\footnotesize\small
    \let\footnoterule\relax
    \let \footnote \thanks
    \setcounter{footnote}{0}
    \null\vfil
    \vskip 60\p@
    \begin{center}
      \setlength{\parskip}{0pt}
      {\scshape\LARGE\textbf{\univname}\par}
      %% TODO: Change all the descriptions to italic like the Thesis one
      \bigskip
      {\LARGE \facname \par}
      {\LARGE \deptname \par}
      {\LARGE \groupname \par}
      \vfill
      \hspace{6mm}\parbox[t][101mm][s]{109mm}{
        \center
        \vfill
        {\huge \bf \@title \par}
        \vfill
        {\Large \textit{by} \textbf\authornames \par}
        \vfill
        {\Large \@date \par}
        \vfill
      }
      \parbox[t][95mm][s]{99mm}{}
      \vfill
      {\Large Project Supervisor\supplural: \supname \par}
      % \ifthenelse{\isempty{\examname}}
      {\Large Second Examiner\examplural: \examname \par}{}
      \vfill
      {\LARGE A project report submitted for the award of }
      {\LARGE MEng Electronic Engineering with Artificial Intelligence \par}
      \vfill
    \end{center}
    \par
    \@thanks
    \vfil\null
  \end{titlepage}
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax-
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
  % \cleardoublepage
}
\newenvironment{abstract}
{
  \btypeout{Abstract Page}
  \thispagestyle{empty}
  \null\vfil\vfil
  \begingroup
  \centering
    \setlength{\parskip}{0pt}
    {\textsc\normalsize \univname \par}
    \bigskip
    {\underline{Abstract} \par}
    \bigskip
    {\textsc\normalsize \facname \par}
    {\textsc\normalsize \deptname \par}
    \bigskip
    {\normalsize\bf \@title \par}
    \medskip
    {\normalsize by \authornames \par}
    \bigskip
  \endgroup
}
{
  \vfil\vfil\null
  % \cleardoublepage
}
\usepackage[nottoc]{tocbibind}         % Put the Lists, Glossary, Biblog and Idx in the contents
\addtocounter{secnumdepth}{1}
\setcounter{tocdepth}{6}
\newcounter{dummy}
\newcommand\addtotoc[1]{
\refstepcounter{dummy}
\addcontentsline{toc}{chapter}{#1}}
\renewcommand\tableofcontents{
\hypersetup{linkcolor={black}}
\btypeout{Table of Contents}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{
          \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}
    \@starttoc{toc}
    \if@restonecol\twocolumn\fi
  \cleardoublepage
}\end{spacing}
}
\renewcommand\listoffigures{
\addtotoc{\listfigurename}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\MakeUppercase\listfigurename}
              {\MakeUppercase\listfigurename}}
    \@starttoc{lof}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\renewcommand\listoftables{
\addtotoc{\listtablename}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{
          \MakeUppercase\listtablename}{\MakeUppercase\listtablename}}
    \@starttoc{lot}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\newcommand\listsymbolname{Definitions and Abbreviations}
\usepackage{longtable}
\newcommand\listofsymbols[2]{
\btypeout{\listsymbolname}
\addtotoc{\listsymbolname}
    \chapter*{\listsymbolname
      \@mkboth{
          \MakeUppercase\listsymbolname}{\MakeUppercase\listsymbolname}}
\begin{longtable}[c]{#1}#2\end{longtable}\par
    % \cleardoublepage
}
\newcommand\addmaterialname{List of Additional Material}
\newcommand\listofaddmaterial{
\addtotoc{\addmaterialname}
\begin{spacing}{1}{
    \setlength{\parskip}{1pt}
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\addmaterialname
      \@mkboth{
          \MakeUppercase\addmaterialname}{\MakeUppercase\addmaterialname}}
    \@starttoc{lom}
    \if@restonecol\twocolumn\fi
    \cleardoublepage
}\end{spacing}
}
\newcommand\addtolom[1]{%
\addtocontents{lom}{\protect\contentsline{chapter}{\protect\numberline{}#1}{}{}}
}
\usepackage{xifthen}
\newcommand\authorshipdeclaration[1]{
\btypeout{Declaration of Authorship}
\addtotoc{Declaration of Authorship}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large\bf Declaration of Authorship \par}\end{center}
I declare that this thesis and the work presented in it is my own and has
been generated by me as the result of my own original research.

I confirm that:

\begin{enumerate}
\item This work was done wholly or mainly while in candidature for a research degree at this University;
\item Where any part of this thesis has previously been submitted for a degree or any other qualification at this University or any other institution, this has been clearly stated;
\item Where I have consulted the published work of others, this is always clearly attributed;
\item Where I have quoted from the work of others, the source is always given. With the exception of such quotations, this thesis is entirely my own work;
\item I have acknowledged all main sources of help;
\item Where the thesis is based on work done by myself jointly with others, I have made clear exactly what was done by others and what I have contributed myself;

\ifthenelse{\isempty{#1}}
{\item None of this work has been published before submission}
{\item Parts of this work have been published as: #1}
\end{enumerate}

\vspace{15.0mm}
\begin{minipage}[t]{0.7\textwidth}
  Signed:..........................................................................
\end{minipage}%
\begin{minipage}[t]{0.3\textwidth}
  Date:..................
\end{minipage}
\vfil\vfil\null
\cleardoublepage
}

\newcommand\acknowledgements[1]{
\btypeout{Acknowledgements}
\addtotoc{Acknowledgements}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large\bf Acknowledgements \par}\end{center}
{\normalsize #1}
\vfil\vfil\null
\cleardoublepage
}
\newcommand\dedicatory[1]{
\btypeout{Dedicatory}
\thispagestyle{plain}
\null\vfil
\vskip 60\p@
\begin{center}{\Large \sl #1}\end{center}
\vfil\null
\cleardoublepage
}
\renewcommand\bibname{References}
\let\oldappendix\appendix % Put full appendix name in ToC
\renewcommand{\appendix}{%
  \oldappendix
  \let\oldacl@pp=\addcontentsline
  \def\addcontentsline##1##2##3{\oldacl@pp{##1}{##2}{\appendixname\space ##3}}
}
\let\oldbackmatter\backmatter
\def\backmatter{%
  \oldbackmatter
  \let\addcontentsline=\oldacl@pp
  }
\endinput
%%
%% End of file `uosprogress.cls'.
