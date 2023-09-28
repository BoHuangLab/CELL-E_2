function Nav() {
  return (
    <nav className={`absolute top-10 min-w-[100%] z-20 hidden md:block`}>
      <ul
        className={`
        py-[4px] px-1 xl:px-2 mx-auto 
        flex justify-center 
        max-w-[30rem] 
        text-xs xl:text-sm text-zinc-300 font-semibold tracking-normal 
        rounded-lg shadow-lg shadow-zinc-800 backdrop-blur bg-black/40 
        ring-1 ring-zinc-600
        `}
      >
        {[
          { title: 'Model', href: '#model' },
          { title: 'Demo 1', href: '#demo1' },
          { title: 'Demo 2', href: '#demo2' },
          { title: 'Authors', href: '#authors' },
          { title: '↑ Top', href: '#top' }
        ].map(item => (
          <li key={item.title}>
            <a
              href={item.href}
              className={`
              relative block px-5 py-2 
              hover:text-zinc-50 
              transition
              `}
            >
              {item.title}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
export default Nav;
