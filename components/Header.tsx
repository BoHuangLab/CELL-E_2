import Canvas from './Canvas';

function Header() {
  return (
    <header
      className={`
        relative px-4 xl:px-32 mx-auto
        max-w-container
      `}
    >
      <div
        className={`
          relative mx-auto
          grid xl:grid-cols-2 xl:grid-rows-1 grid-cols-1 grid-rows-2
          max-w-[100rem]
          border-l-[1px] border-zinc-800 border-r-[1px]
          z-10
        `}
      >
        <div
          className={`
            px-4 mt-24 mb-4 xl:px-32 xl:mt-72 xl:mb-16
            col-span-1
          `}
        >
          <h1
            className={`
              text-7xl xl:text-9xl text-white font-bold tracking-tight
            `}
          >
            CELL-E <span className={`gradient`}>2</span>
          </h1>
          <h2
            className={`
              text-[#888] text-xl xl:text-2xl my-8 ml-1 leading-7 xl:leading-10 whitespace-pre
            `}
          >
            {`Text-to-Image Models for \nProtein Localization Prediction`}
          </h2>
        </div>

        <div
          className={`
            col-span-1 bg-zinc-900 border-l-[1px] border-b-[1px] border-zinc-800
          `}
        >
          <Canvas />
        </div>

        <svg
          className={`
            absolute inset-0 h-full w-full text-zinc-800 hover:text-zinc-800/75 -z-10 transition
          `}
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <pattern
              id="pricing-pattern"
              width="32"
              height="64"
              patternUnits="userSpaceOnUse"
              x="50%"
              y="100%"
              patternTransform="translate(0 -1)"
            >
              <path d="M0 64V.5H32" fill="none" stroke="currentColor"></path>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#pricing-pattern)"></rect>
        </svg>
      </div>

      <style jsx>{`
        .gradient {
          background: linear-gradient(
            330deg,
            #b100ff 20%,
            #ff0080 40%,
            #b100ff 60%,
            #ff0080 80%
          );
          background-size: 100% auto;

          color: #000;
          background-clip: text;
          text-fill-color: transparent;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;

          text-shadow: 0px 0px 20px #ff008040;

          animation: shine 15s infinite;
        }
      `}</style>
    </header>
  );
}

export default Header;
