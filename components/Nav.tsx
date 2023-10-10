import { useCallback, useEffect, useState } from 'react';

function useDebouncedScrollPosition(delay = 200) {
  const [scrollPosition, setScrollPosition] = useState(0);

  const updatePosition = useCallback(() => {
    setScrollPosition(window.pageYOffset);
  }, []);

  useEffect(() => {
    let timeoutId: null | number;

    const handleScroll = () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }

      timeoutId = window.setTimeout(updatePosition, delay);
    };

    window.addEventListener('scroll', handleScroll);
    updatePosition();

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      window.removeEventListener('scroll', handleScroll);
    };
  }, [delay, updatePosition]);

  return scrollPosition;
}
function Nav() {
  return (
    <nav className={`absolute top-10 min-w-[100%] z-20 hidden md:block`}>
      <ul
        className={`
          py-[4px] px-1 xl:px-2 mx-auto
          flex justify-center
          max-w-[30rem] rounded-lg
          text-zinc-300 text-xs xl:text-sm font-semibold tracking-normal 
          bg-black/30 backdrop-blur
          ring-1 ring-zinc-600
          shadow-lg shadow-zinc-800
        `}
      >
        {[
          { title: 'Model', href: '#model' },
          { title: 'Demo 1', href: '#demo1' },
          { title: 'Demo 2', href: '#demo2' },
          { title: 'Design', href: '#design' },
          { title: 'Authors', href: '#authors' }
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
