/* wmdel <window id>: asks a window to close the way a window manager's close button does (WM_DELETE_WINDOW) */
#include <X11/Xlib.h>
#include <stdlib.h>
#include <string.h>
int main (int c, char** v)
{
    Display* d = XOpenDisplay (0); if (! d || c < 2) return 1;
    if (strcmp (v[1], "intern") == 0)      /* make the atoms exist before the app starts: without a window manager nobody else does, and the app only looks them up */
    {
        XInternAtom (d, "WM_PROTOCOLS", False); XInternAtom (d, "WM_DELETE_WINDOW", False); XInternAtom (d, "WM_TAKE_FOCUS", False); XInternAtom (d, "_NET_WM_PING", False);
        XSync (d, False); XCloseDisplay (d); return 0;
    }
    Window w = strtoul (v[1], 0, 0); XEvent e; memset (&e, 0, sizeof (e));
    e.xclient.type = ClientMessage; e.xclient.window = w; e.xclient.message_type = XInternAtom (d, "WM_PROTOCOLS", False); e.xclient.format = 32;
    e.xclient.data.l[0] = (long) XInternAtom (d, "WM_DELETE_WINDOW", False); e.xclient.data.l[1] = CurrentTime;
    XSendEvent (d, w, False, NoEventMask, &e); XFlush (d); XCloseDisplay (d); return 0;
}
