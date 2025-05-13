function sendExperienceToPython(expStruct)
    persistent sock
    if isempty(sock)
        import py.zmq.Context py.zmq.REQ
        ctx  = Context();
        sock = ctx.socket(REQ);
        sock.connect('tcp://127.0.0.1:5555');
    end

    % 메시지 인코딩
    msg     = struct('type', expStruct.type, 'exp', expStruct);
    jsonReq = jsonencode(msg);

    % send → 반드시 recv
    sock.send_string(py.str(jsonReq));
    jsonResp = char( sock.recv_string() );
    % (응답 내용 확인이 필요 없으면 여기서 끝)
end
