function Footer() {
  return (
    <footer
      className={`
        mx-auto pt-32 pb-4 px-4 xl:px-8 
        w-full max-w-container 
        bg-[#f6f9fc]
      `}
    >
      <div className={`py-10 border-t border-slate-900/5`}>
        <p className={`mt-5 text-center text-sm leading-6 text-slate-500`}>
          All rights reserved.{' '}
          <a
            className="text-slate-700"
            rel="noopener noreferrer"
            target="_blank"
            href="https://github.com/BoHuangLab/CELL-E_2/tree/gh-pages"
          >
            Site code
          </a>
          .
        </p>
      </div>
    </footer>
  );
}

export default Footer;
