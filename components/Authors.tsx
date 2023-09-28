function Authors() {
  return (
    <div
      id="authors"
      className={`
        mx-auto px-4 lg:px-32 
        max-w-container min-h-32 h-64 lg:min-h-24 lg:h-32
        bg-black
      `}
    >
      <div
        className={`
          mx-auto 
          flex flex-col justify-center 
          max-w-[100rem] h-64 lg:h-32 
          bg-zinc-900 
          border-l-[1px] border-r-[1px] border-solid border-zinc-800
        `}
      >
        <h4
          className={`
            pb-3 lg:pb-6 
            flex justify-center 
            uppercase text-indigo-400 text-xs font-semibold tracking-wider
          `}
        >
          Authors
        </h4>

        <ul
          className={`
            mx-auto 
            flex flex-col lg:flex-row items-center lg:justify-center
            tracking-[0.2px] text-md text-zinc-200 font-medium
          `}
        >
          {[
            {
              name: 'Emaad Khwaja',
              href: 'https://www.linkedin.com/in/emaad/'
            },
            {
              name: 'Yun S. Song',
              href: 'http://people.eecs.berkeley.edu/~yss/'
            },
            { name: 'Aaron Agarunov', href: 'https://agarun.com/' },
            { name: 'Bo Huang', href: 'http://huanglab.ucsf.edu/' }
          ].map(item => {
            return (
              <li key={item.name} className="my-3 lg:my-0">
                <a
                  href={item.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`
                    py-2 px-3 mx-2 
                    h-11
                    hover:text-indigo-200 
                    rounded-lg shadow 
                    border-solid border-[1px] border-zinc-600 hover:border-indigo-600 
                  `}
                >
                  {item.name}
                </a>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

export default Authors;
