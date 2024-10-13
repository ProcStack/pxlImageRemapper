
#
#    HoverButton Color Themes
#    By Kevin Edzenga; 2024
#
#  -- -- -- -- -- -- --
#
#   Standalone Unit Test -
#     No unit test for this file
#       This file is used by HoverButtonWidget
#
#   Currently available color themes:
#    `DEFAULT`, `GREEN`, `YELLOW`, `RED`, `BLUE`
#      `DEFAULT` is a Blue/Grey Theme
#    
#  You can add more themes below
#    There is a template at the bottom
#

class pxlColors():
  THEMES = {
    # -- -- -- -- -- -- -- -- -- -- -- -- --
    # -- Default Theme;  Blue'ish Grey -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    "DEFAULT": {
      "BASE": {
        "text": "#ffffff",
        "bg": "#353535",
        "brd": "#808080"
      },
      "HOVER": {
        "text": "#eff6ff",
        "bg": "#404550",
        "brd": "#5599cc"
      },
      "DOWN": {
        "text": "#dddddd",
        "bg": "#303030",
        "brd": "#707070"
      }
    },

    # -- -- -- -- -- -- -- -- -- --
    # -- Accept Theme;  Green -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- --
    "GREEN": {
      "BASE": {
        "text": "#ffffff",
        "bg": "#405040",
        "brd": "#808080"
      },
      "HOVER": {
        "text": "#dfffdf",
        "bg": "#557055",
        "brd": "#70cc70"
      },
      "DOWN": {
        "text": "#ceeece",
        "bg": "#303530",
        "brd": "#60bb60"
      }
    },

    # -- -- -- -- -- -- -- -- -- --
    # -- Info Theme;  Yellow  -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- --
    "YELLOW": {
      "BASE": {
        "text": "#ffffff",
        "bg": "#505040",
        "brd": "#808080"
      },
      "HOVER": {
        "text": "#fffcff",
        "bg": "#707045",
        "brd": "#cccc55"
      },
      "DOWN": {
        "text": "#eeecee",
        "bg": "#353035",
        "brd": "#bbbb45"
      }
    },

    # -- -- -- -- -- -- -- -- -- --
    # -- Warning Theme;  Red  -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- --
    "RED": {
      "BASE": {
        "text": "#ffffff",
        "bg": "#553535",
        "brd": "#808080"
      },
      "HOVER": {
        "text": "#ffdfdf",
        "bg": "#704550",
        "brd": "#cc5065"
      },
      "DOWN": {
        "text": "#eedddd",
        "bg": "#352525",
        "brd": "#705050"
      }
    },

    # -- -- -- -- -- -- -- -- -- --
    # -- Confirm Theme;  Blue  -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- --
    "BLUE": {
      "BASE": {
        "text": "#ffffff",
        "bg": "#353560",
        "brd": "#808080"
      },
      "HOVER": {
        "text": "#dfdfff",
        "bg": "#455070",
        "brd": "#5065dd"
      },
      "DOWN": {
        "text": "#ddddee",
        "bg": "#252535",
        "brd": "#505070"
      }
    }

    # -- -- -- -- -- -- -- -- -- --
    # -- New Theme;  ****  -- -- -- --
    # -- -- -- -- -- -- -- -- -- -- -- --
    # "NEW_THEME": {
    #   "BASE": {
    #     "text": "#",
    #     "bg": "#",
    #     "brd": "#"
    #   },
    #   "HOVER": {
    #     "text": "#",
    #     "bg": "#",
    #     "brd": "#"
    #   },
    #   "DOWN": {
    #     "text": "#",
    #     "bg": "#",
    #     "brd": "#"
    #   }
    # }

  }

# -- -- --

# Busted Unit Test
if __name__ == "__main__":
  # Sing along with me!
  #   https://www.youtube.com/watch?v=aNddW2xmZp8
  print("Oooooohhh,  -I'd love to be an oscar mayer weiner...")
  print("...That is what I'd truly like to beee-eee-eee!!")
  print("Caauussee- if I were an oscar mayer weiner...")
  print("...Everyone would be in love with meeeeee!!")
