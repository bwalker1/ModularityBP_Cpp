if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
imap <silent> <Plug>IMAP_JumpBack =IMAP_Jumpfunc('b', 0)
imap <silent> <Plug>IMAP_JumpForward =IMAP_Jumpfunc('', 0)
map! <F12> :w:make %< 
map! <F7> :w:!gnuplot "%" -persist 
map! <F4> :wa:make
map! <F2> :tabp
map! <F3> :tabn
map! <F1> :wa:make
vmap <NL> <Plug>IMAP_JumpForward
nmap <NL> <Plug>IMAP_JumpForward
omap <NL> :w:make %< 
map  :w:!latex "%"
nmap gx <Plug>NetrwBrowseX
map gg :Tabswitch
map ww w
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#NetrwBrowseX(expand("<cWORD>"),0)
vmap <silent> <Plug>IMAP_JumpBack `<i=IMAP_Jumpfunc('b', 0)
vmap <silent> <Plug>IMAP_JumpForward i=IMAP_Jumpfunc('', 0)
vmap <silent> <Plug>IMAP_DeleteAndJumpBack "_<Del>i=IMAP_Jumpfunc('b', 0)
vmap <silent> <Plug>IMAP_DeleteAndJumpForward "_<Del>i=IMAP_Jumpfunc('', 0)
nmap <silent> <Plug>IMAP_JumpBack i=IMAP_Jumpfunc('b', 0)
nmap <silent> <Plug>IMAP_JumpForward i=IMAP_Jumpfunc('', 0)
map <F12> :w:make %< 
map <F9>:!latex "%":Bibtex:!latex "%"
map <F9> :w:!latex "%"
map <F7> :w:!gnuplot "%" -persist 
map <F4> :wa:make
map <F2> :tabp
map <F3> :tabn
nmap <F6> yypki//{{{jj%o//}}}zm
nmap <silent> <F5> :QFix
nnoremap <silent> <F8> :TlistToggle
map <F1> :wa:make
imap <NL> <Plug>IMAP_JumpForward
cmap <NL> :w:make %< 
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set backspace=indent,eol,start
set backup
set backupdir=~/tmp/vibackup
set directory=~/tmp/vibackup
set fileencodings=ucs-bom,utf-8,gb18030,gbk,gb2312,big5,euc-jp,euc-kr,latin1,cp936
set grepprg=grep\ -nH\ $*
set helplang=en
set history=1000
set incsearch
set listchars=tab:.\ ,trail:'
set printoptions=paper:letter
set ruler
set runtimepath=~/.vim,/var/lib/vim/addons,/usr/share/vim/vimfiles,/usr/share/vim/vim73,/usr/share/vim/vimfiles/after,/var/lib/vim/addons/after,~/.vim/after
set shiftwidth=4
set showmatch
set showtabline=2
set smartindent
set suffixes=.bak,~,.swp,.o,.info,.aux,.log,.dvi,.bbl,.blg,.brf,.cb,.ind,.idx,.ilg,.inx,.out,.toc
set tabline=%!MyTabLine()
set tabstop=4
set termencoding=utf-8
set visualbell
" vim: set ft=vim :
