function Footer() {
  return (
    <footer
      className={`
        mx-auto pt-32 px-4 xl:px-8 
        w-full max-w-container 
        bg-[#f6f9fc]
      `}
    >
      <div className={`py-10 border-t border-slate-900/5`}>
        <p className={`mt-5 text-center text-sm leading-6 text-slate-500`}>
          All rights reserved.
        </p>
      </div>
    </footer>
  );
}

export default Footer;
