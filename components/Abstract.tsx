import { useState } from 'react';
import { abstract } from '../app/data/text';

function Abstract() {
  const [showFullText, setShowFullText] = useState(false);

  return (
    <div
      className={`
        px-4
        flex flex-col items-center 
        bg-gray-100
      `}
    >
      <div
        className={`
          mx-auto 
          flex justify-center 
          w-full max-w-[100rem] 
          border-l-[1px] border-r-[1px] border-dashed border-zinc-300
        `}
      >
        <div
          className={`
            pt-14 pb-12 
            mx-auto 
            max-w-[100rem] 
            border-l-[1px] border-r-[1px] border-dotted border-zinc-300
          `}
        >
          <p
            className={`
              px-6 md:px-4 max-w-[50rem] text-[#0a2540] text-2xl text-center leading-9 font-semibold
              transition ease-in-out
              ${showFullText ? `px-2 md:px-14` : ''}
            `}
          >
            {abstract[0]}.
            {showFullText ? (
              <span className="font-normal">
                {abstract.slice(1).join('. ')}
              </span>
            ) : (
              <button
                className={`
                  mx-1 px-1 rounded-lg bg-zinc-200 hover:bg-zinc-100 border-solid border-[1px] border-zinc-300 hover:border-zinc-100 font-bold text-zinc-400
                `}
                onClick={() => setShowFullText(true)}
                title="Read More"
              >
                …
              </button>
            )}
          </p>
          <div className={`flex flex-wrap justify-center`}>
            <a
              className={`
                m-3 mb-0 flex content-center items-center rounded-lg mt-6 py-5 px-4 h-11 bg-indigo-600 text-white  ring-indigo-800 hover:bg-[#f6f9fc] hover:text-black hover:border-zinc-300 border-[1px] transition
              `}
              href="https://openreview.net/forum?id=YSMLVffl5u"
              target="_blank"
              rel="noopener noreferrer"
            >
              View Paper <span className="text-3xl ml-2 mt-[-2px]">{'›'}</span>
            </a>
            <a
              className={`
                m-3 mb-0 flex content-center items-center rounded-lg mt-6 py-5 px-4 h-11 bg-[#f6f9fc] border-solid border-[1px] text-zinc-900 hover:bg-zinc-100 font-medium ring-zinc-300
              `}
              href="https://github.com/BoHuangLab/CELL-E_2"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            <a
              className={`
                m-3 mb-0 flex content-center items-center rounded-lg mt-6 py-5 px-4 h-11 bg-[#f6f9fc] border-solid border-[1px] text-zinc-900 hover:bg-zinc-100 font-medium ring-zinc-300
              `}
              href="https://huggingface.co/HuangLab/CELL-E_2_HPA_Finetuned_480"
              target="_blank"
              rel="noopener noreferrer"
            >
              HuggingFace
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Abstract;
